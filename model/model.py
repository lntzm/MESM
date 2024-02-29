# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from utils import split_and_pad, split_expand_and_pad
from utils import sample_outclass_neg, sample_inclass_neg, inverse_sigmoid
from .transformer import T2V_TransformerEncoderLayer, T2V_TransformerEncoder
from .text_encoder import CLIPTextEncoder, GloveTextEncoder


class MESM(nn.Module):
    """ This is the Moment-DETR module that performs moment localization. """

    def __init__(
        self, text_encoder, enhance_encoder, t2v_encoder, transformer, 
        vid_position_embed, txt_position_embed, txt_dim, vid_dim,
        num_queries, input_dropout, 
        aux_loss=False, max_video_l=75, max_words_l=32, 
        normalize_txt=True, use_txt_pos=False,
        span_loss_type="l1", n_input_proj=2,
        rec_fw=False, vocab_size=1111,
        rec_ss=False, num_recss_layers=2,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)
        self.enhance_encoder = enhance_encoder      # FW-MESM
        self.t2v_encoder = t2v_encoder              # Aligner
        self.transformer = transformer              # DETR
        self.vid_position_embed = vid_position_embed
        self.txt_position_embed = txt_position_embed
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_video_l = max_video_l
        self.max_words_l = max_words_l
        self.normalize_txt = normalize_txt
        span_pred_dim = 2 if span_loss_type == "l1" else max_video_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 2)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.global_rep_token = nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = nn.Parameter(torch.randn(hidden_dim))

        # frame-word level masked language modeling (reconstruction)
        self.rec_fw = rec_fw
        if isinstance(self.text_encoder, CLIPTextEncoder):
            num_classes = vocab_size + 3
        elif isinstance(self.text_encoder, GloveTextEncoder) or self.text_encoder is None:
            num_classes = vocab_size + 1
        else:
            raise NotImplementedError
        
        # frame-word level reconstruction (FW-MESM)
        if rec_fw:
            self.masked_token = nn.Parameter(torch.zeros(txt_dim).float(), requires_grad=True)
            self.unknown_token = nn.Parameter(torch.zeros(txt_dim).float(), requires_grad=True)
            self.output_txt_proj = nn.Sequential(*[
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=True),
                nn.Linear(hidden_dim, num_classes),
            ])
        
        # segment-sentence level reconstruction (SS-MESM)
        self.rec_ss = rec_ss
        if rec_ss:
            self.ss_reconstructor = SegSenRecon(    # SS-MESM
                input_dropout=input_dropout, hidden_dim=hidden_dim, nhead=transformer.nhead,
                num_layers=num_recss_layers, dim_feedforward=transformer.dim_feedforward,
                dropout=transformer.dropout, activation=transformer.activation,
                normalize_before=transformer.normalize_before
            )
            ## ablation 1
            # self.mix_txt_proj = LinearLayer(hidden_dim*2, hidden_dim, layer_norm=True, dropout=0, relu=False)
            # self.mix_txt_proj = nn.Linear(hidden_dim*2, hidden_dim)
        
    def CLIP_encode_text(self, words_id, words_mask, device):
        if device == torch.device(type='cpu'):
            self.text_encoder.to("cuda")
            words_id = words_id.to("cuda")
        txt_feat = self.text_encoder(words_id)
        words_feat = txt_feat["last_hidden_state"].to(torch.float32)

        if device == torch.device(type='cpu'):
            words_id = words_id.to("cpu")
            words_feat = words_feat.to("cpu")
            

        words_feat = words_feat[:, :self.max_words_l, :]
        words_id = words_id[:, :self.max_words_l]
        words_mask = words_mask[:, :self.max_words_l]
        words_feat.masked_fill_(words_mask.unsqueeze(-1)==False, 0)
        # ## ablation 1
        # sentence_feat = txt_feat["pooler_output"].to(torch.float32)

        # ## ablation 2
        sentence_feat = words_feat.sum(dim=1) / words_mask.sum(dim=1).unsqueeze(-1)

        ## ablation 3
        # sentence_feat = words_feat.max(dim=1)[0]
        if device == torch.device(type='cpu'):
            sentence_feat = sentence_feat.to("cpu")
        
        if self.normalize_txt:
            words_feat = F.normalize(words_feat, dim=-1, p=2, eps=1e-5)
            sentence_feat = F.normalize(sentence_feat, dim=-1, p=2, eps=1e-5)
        
        return words_feat, sentence_feat, words_id, words_mask
    
    def GloVe_encode_text(self, words_id, words_mask):
        words_feat = self.text_encoder(words_id)
        words_feat.masked_fill_(words_mask.unsqueeze(-1)==False, 0)
        sentence_feat = words_feat.sum(dim=1) / words_mask.sum(dim=1).unsqueeze(-1)
        if self.normalize_txt:
            words_feat = F.normalize(words_feat, dim=-1, p=2, eps=1e-5)
            sentence_feat = F.normalize(sentence_feat, dim=-1, p=2, eps=1e-5)
        return words_feat, sentence_feat

    def post_process_text(self, words_feat):
        if self.normalize_txt:
            words_feat = F.normalize(words_feat, dim=-1, p=2, eps=1e-5)
        words_mask = words_feat.sum(dim=-1) != 0
        sentence_feat = words_feat.sum(dim=1) / words_mask.sum(dim=1).unsqueeze(-1)
        if self.normalize_txt:
            sentence_feat = F.normalize(sentence_feat, dim=-1, p=2, eps=1e-5)
        return words_feat, words_mask, sentence_feat

    def forward(self, video_feat, video_mask, words_id, words_mask, words_weight, num_clips, **kwargs):
        if isinstance(self.text_encoder, CLIPTextEncoder):
            words_feat, sentence_feat, words_id, words_mask = \
                self.CLIP_encode_text(words_id, words_mask, device=words_id.device)
        elif isinstance(self.text_encoder, GloveTextEncoder):
            words_feat, sentence_feat = self.GloVe_encode_text(words_id, words_mask)
        elif self.text_encoder is None:
            words_feat, words_mask, sentence_feat = self.post_process_text(words_id)
        else:
            raise NotImplementedError

        batch_size = video_feat.shape[0]
        projed_video_feat = self.input_vid_proj(video_feat)
        projed_words_feat = self.input_txt_proj(words_feat)
        vid_position = self.vid_position_embed(projed_video_feat, video_mask)  # (bsz, L_vid, d)
        if self.use_txt_pos:
            txt_position = self.txt_position_embed(projed_words_feat)
        else:
            txt_position = torch.zeros_like(projed_words_feat)


        if self.rec_fw:
            enhanced_video_feat = self.enhance_encoder(
                projed_words_feat, projed_video_feat, 
                src_txt_key_padding_mask=~words_mask, pos_txt=txt_position,
                src_vid_key_padding_mask=~video_mask, pos_vid=vid_position
            )
        else:
            enhanced_video_feat = projed_video_feat

        if self.rec_ss:
            if kwargs['dataset_name'] in ["charades", "charades-cg", "charades-cd", "tacos"]:
                batched_vid = video_feat
                # batched_vid = enhanced_video_feat
                batched_vid_mask = video_mask
                batched_vid_position = vid_position
            elif kwargs['dataset_name'] in ["qvhighlights"]:
                video_length = torch.stack([i.sum() for i in torch.split(video_mask, num_clips.tolist())]).long()
                unpadded_video_feat = video_feat[video_mask]
                batched_vid, batched_vid_mask = split_expand_and_pad(video_length, num_clips, unpadded_video_feat)
                ## ablation 1
                batched_vid_position = self.vid_position_embed(batched_vid, batched_vid_mask)
            else:
                raise NotImplementedError
            
            batched_sent, batched_sent_mask = split_expand_and_pad(num_clips, num_clips, sentence_feat)

            batched_vid = self.input_vid_proj(batched_vid)
            batched_sent = self.input_txt_proj(batched_sent)

            recon_feat, projed_recon_feat = self.ss_reconstructor(
                batched_vid, batched_vid_mask, batched_sent, batched_sent_mask,
                num_clips, batched_vid_position
            )

            # ## ablation 1
            # recon_feat = recon_feat.unsqueeze(1).repeat(1,projed_words_feat.shape[1],1)
            # expanded_words_feat = torch.cat([recon_feat, projed_words_feat], dim=2)
            # expanded_words_feat = self.mix_txt_proj(expanded_words_feat)
            # expanded_words_mask = words_mask
            # expanded_words_feat.masked_fill_(expanded_words_mask.unsqueeze(dim=2)==0, 0.)

            ## ablation 2
            expanded_words_feat = torch.cat([recon_feat.unsqueeze(1), projed_words_feat], dim=1)
            recon_mask = torch.ones([batch_size, 1], dtype=torch.bool, device=recon_feat.device)
            expanded_words_mask = torch.cat([recon_mask, words_mask], dim=1)
        else:
            expanded_words_feat = projed_words_feat
            expanded_words_mask = words_mask

        # TODO should we remove or use different positional embeddings to the src_txt?
        if self.use_txt_pos:
            expanded_txt_position = self.txt_position_embed(expanded_words_feat)
        else:
            expanded_txt_position = torch.zeros_like(expanded_words_feat)

        encoded_video_feat = self.t2v_encoder(
            expanded_words_feat, enhanced_video_feat, 
            src_txt_key_padding_mask=~expanded_words_mask, pos_txt=expanded_txt_position,
            src_vid_key_padding_mask=~video_mask, pos_vid=vid_position
        )

        # for global token
        global_token = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(batch_size, 1, 1)
        global_token_pos = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(batch_size, 1, 1)

        hs, reference, memory, memory_global = self.transformer(
            encoded_video_feat, ~video_mask, 
            self.query_embed.weight, vid_position, 
            global_token, global_token_pos
        )

        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        # outputs_coord = tmp
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        
        # if kwargs["dataset_name"] in ["tacos"]:
        #     neg_index = sample_inclass_neg(num_clips, kwargs['norm_moment'])
        # else:
        #     neg_index = sample_outclass_neg(num_clips)
        # if len(tuple(set(kwargs['video_id']))) > 1:
        #     pass
        neg_index = sample_outclass_neg(num_clips)
        neg_expanded_words_feat = expanded_words_feat[neg_index]
        neg_expanded_words_mask = expanded_words_mask[neg_index]
        neg_expanded_txt_position = expanded_txt_position[neg_index]
        if self.rec_ss:
            neg_words_feat = neg_expanded_words_feat[:, 1:, :]
            neg_words_mask = neg_expanded_words_mask[:, 1:]
            neg_txt_position = neg_expanded_txt_position[:, 1:, :]
        else:
            neg_words_feat = neg_expanded_words_feat
            neg_words_mask = neg_expanded_words_mask
            neg_txt_position = neg_expanded_txt_position
        neg_vid_position = vid_position.clone()  # since it does not use actual content
        # neg_projed_video_feat = projed_video_feat

        # # final ablation for FWSM, self.rec_fw should set to False
        # neg_enhanced_video_feat = self.enhance_encoder(
        #     neg_words_feat, projed_video_feat, 
        #     src_txt_key_padding_mask=~neg_words_mask, pos_txt=neg_txt_position,
        #     src_vid_key_padding_mask=~video_mask, pos_vid=neg_vid_position
        # )
        if self.rec_fw:
            neg_enhanced_video_feat = self.enhance_encoder(
                neg_words_feat, projed_video_feat, 
                src_txt_key_padding_mask=~neg_words_mask, pos_txt=neg_txt_position,
                src_vid_key_padding_mask=~video_mask, pos_vid=neg_vid_position
            )
        else:
            neg_enhanced_video_feat = projed_video_feat
        
        neg_encoded_video_feat = self.t2v_encoder(
            neg_expanded_words_feat, neg_enhanced_video_feat, 
            src_txt_key_padding_mask=~neg_expanded_words_mask, pos_txt=neg_expanded_txt_position,
            src_vid_key_padding_mask=~video_mask, pos_vid=neg_vid_position
        )
        _, _, neg_memory, neg_memory_global = self.transformer(
            neg_encoded_video_feat, ~video_mask,
            self.query_embed.weight, neg_vid_position,
            global_token, global_token_pos
        )

        saliency_scores = torch.sum(self.saliency_proj1(memory) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim)
        neg_saliency_scores = torch.sum(self.saliency_proj1(neg_memory) * self.saliency_proj2(neg_memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim)

        if self.aux_loss:
            aux_outputs = [{'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        if self.rec_fw and kwargs["is_training"]:
            unknown_mask = kwargs["unknown_mask"]
            unknowned_words_feat = self._replace_unknown(projed_words_feat, unknown_mask, self.unknown_token, proj=True)
            clip_mask = kwargs["clip_mask"]
            selected_video_feat = projed_video_feat[clip_mask]
            selected_length = clip_mask.sum(dim=1)
            merged_clip_feat, merged_clip_mask = split_and_pad(selected_length, selected_video_feat)

            masked_words_feat, masked_words_loc = self._mask_words(unknowned_words_feat, words_mask, self.masked_token, proj=True, weight=words_weight)
            # if self.rec_ss:
            #     expanded_masked_words_feat = torch.cat([recon_feat.unsqueeze(1), masked_words_feat], dim=1)
            # else:
            #     expanded_masked_words_feat = masked_words_feat
            # ## ablation 1
            # merged_clip_position = self.vid_position_embed(merged_clip_feat, merged_clip_mask)

            ## ablation 2
            selected_vid_position = vid_position[clip_mask]
            merged_clip_position, _ = split_and_pad(selected_length, selected_vid_position)

            recfw_out = self.enhance_encoder(
                merged_clip_feat, masked_words_feat,
                src_txt_key_padding_mask=~merged_clip_mask, pos_txt=merged_clip_position,
                src_vid_key_padding_mask=~words_mask, pos_vid=txt_position, is_MLM=True
            )
            recfw_words_logit = self.output_txt_proj(recfw_out)
        
        out = {
            "pred_logits": outputs_class[-1],
            "pred_spans": outputs_coord[-1],
            "saliency_scores": saliency_scores,
            "neg_saliency_scores": neg_saliency_scores,
        }
        if self.aux_loss:
            out.update({"aux_outputs": aux_outputs})
        if self.rec_ss:
            out.update({
                "projed_video_feat": projed_video_feat,
                "recon_feat": recon_feat,
                "projed_recon_feat": projed_recon_feat,
                "expanded_words_feat": expanded_words_feat,
                "expanded_words_mask": expanded_words_mask,
                "enhanced_video_feat": enhanced_video_feat,
                "projed_words_feat": projed_words_feat,
            })
        if self.rec_fw and kwargs["is_training"]:
            out.update({
                # "words_feat": words_feat,
                "words_mask": words_mask,
                "recfw_words_logit": recfw_words_logit,
            })

        return out

    def _mask_words(self, src_txt, src_txt_mask, masked_token, proj=True, weight=None):
        masked_token = masked_token.unsqueeze(0).unsqueeze(0)
        if proj:
            masked_token = self.input_txt_proj(masked_token)

        words_length = src_txt_mask.count_nonzero(dim=1)
        masked_words = torch.zeros_like(src_txt_mask)
        if weight is not None:
            weight = F.normalize(weight.float(), dim=1, p=1)
        for i, l in enumerate(words_length):
            l = int(l)
            if l <= 1:
                continue
            num_masked_words = max(l // 3, 1)
            p = weight[i, :l].numpy() if weight is not None else None
            
            choices = np.random.choice(np.arange(0, l), num_masked_words, replace=False, p=p)
            masked_words[i, choices] = 1
        
        masked_words_vec = src_txt.new_zeros(*src_txt.size()) + masked_token
        masked_words_vec.masked_fill_(masked_words.unsqueeze(-1) == 0, 0)
        masked_src_txt = src_txt.masked_fill(masked_words.unsqueeze(-1) == 1, 0) + masked_words_vec

        return masked_src_txt, masked_words
    
    def _replace_unknown(self, words_feat, unknown_mask, masked_token, proj=True):
        masked_token = masked_token.unsqueeze(0).unsqueeze(0)
        if proj:
            masked_token = self.input_txt_proj(masked_token)
        # words_feat[unknown_mask] = masked_token
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + masked_token
        masked_words_vec.masked_fill_(unknown_mask.unsqueeze(-1) == 0, 0)
        replaced_words_feat = words_feat.masked_fill(unknown_mask.unsqueeze(-1) == 1, 0) + masked_words_vec
        return replaced_words_feat


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class SegSenRecon(nn.Module):
    def __init__(self, input_dropout, hidden_dim=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.masked_sent_token = nn.Parameter(torch.zeros(hidden_dim).float(), requires_grad=True)
        # self.masked_vid_token = torch.nn.Parameter(torch.zeros(hidden_dim).float(), requires_grad=True)

        recon_trans_layer = T2V_TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.recon_trans = T2V_TransformerEncoder(
            recon_trans_layer, num_layers, encoder_norm
        )

        self.output_sent_proj = nn.Sequential(*[
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=True),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=False)
        ])
        # self.mix_txt_proj = nn.Sequential(*[
        #     LinearLayer(2*hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=False)
        # ])
        # self.output_vid_proj2 = nn.Sequential(*[
        #     LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2]),
        #     LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
        #     LinearLayer(hidden_dim, vid_dim + aud_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0])
        # ][3 - n_input_proj:])

    def forward(self, batched_vid, batched_vid_mask, batched_sent, batched_sent_mask, num_clips, vid_position):
        masked_sent_tokens, masked_sent_loc = self._sequence_mask_sent(batched_sent, self.masked_sent_token, num_clips)
        masked_sent_tokens = masked_sent_tokens.permute(1, 0, 2)
        batched_vid = batched_vid.permute(1, 0, 2)
        if vid_position is not None:
            vid_position = vid_position.permute(1, 0, 2)

        recon_sent_tokens = self.recon_trans(
            src_txt=batched_vid, src_vid=masked_sent_tokens,
            src_txt_mask=None, src_txt_key_padding_mask=~batched_vid_mask,
            src_vid_mask=None, src_vid_key_padding_mask=~batched_sent_mask,
            ## ablation 1
            # pos_txt=vid_position, pos_vid=None

            # ## ablation 2
            pos_txt=None, pos_vid=None
        )

        recon_sent_tokens = recon_sent_tokens.permute(1, 0, 2)
        recon_feat = F.normalize(recon_sent_tokens[masked_sent_loc])
        recon_feat_proj = self.output_sent_proj(recon_feat)
        return recon_feat, recon_feat_proj

    def _sequence_mask_sent(self, batched_sent, masked_token, num_clips):
        masked_token = masked_token.unsqueeze(0).unsqueeze(0)
        mask = []
        for num in num_clips:
            index = torch.arange(num, device=num_clips.device)
            sequence_mask = torch.zeros([num, batched_sent.shape[1]], device=batched_sent.device, dtype=torch.bool)
            sequence_mask[index, index] = True
            mask.append(sequence_mask)
        mask = torch.cat(mask)
        masked_vec = batched_sent.new_zeros(*batched_sent.size()) + masked_token
        masked_vec.masked_fill_(mask.unsqueeze(-1)==False, 0)
        masked_sent = batched_sent.masked_fill(mask.unsqueeze(-1), 0) + masked_vec

        return masked_sent, mask

