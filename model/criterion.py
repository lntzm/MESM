import torch
from torch import nn
import torch.nn.functional as F

from utils import generalized_temporal_iou, span_cxw_to_xx
from utils import accuracy


class Criterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, losses,
                 eos_coef, span_loss_type, max_video_l,
                 rank_coef, use_triplet,
                 saliency_margin=1,
                 multi_clip=False,
                 gamma=0.9, recss_tau=0.5):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.span_loss_type = span_loss_type
        self.max_video_l = max_video_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

        self.rank_coef = rank_coef
        self.use_triplet = use_triplet
        
        # addtional losses
        self.rec_ss = "rec_ss" in losses
        self.rec_fw = "rec_fw" in losses
        self.multi_clip = multi_clip
        self.gamma = gamma
        self.recss_tau = recss_tau
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        # assert 'pred_spans' in outputs
        # targets = targets["norm_span"]
        # idx = self._get_src_permutation_idx(indices)
        if self.multi_clip:
            idx = self._get_src_permutation_idx(indices)
            src_spans = outputs['pred_spans'][idx]
            tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets["norm_span"], indices)], dim=0)  # (#spans, 2)
            tgt_moments = span_cxw_to_xx(tgt_spans)
        else:
            batch_idx = torch.arange(indices.shape[0], device=indices.device)
            src_spans = outputs['pred_spans'][batch_idx, indices[:, 0], :]  # (#spans, max_v_l * 2)
            tgt_spans = targets["norm_span"]
            tgt_moments = targets["norm_moment"]
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), tgt_moments))
        else:  # ce
            raise NotImplementedError
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_video_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        # idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        if self.multi_clip:
            idx = self._get_src_permutation_idx(indices)
        else:
            batch_idx = torch.arange(indices.shape[0], device=indices.device)
            idx = (batch_idx, indices[:, 0])

        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        # if "pos_idx" not in targets:
        #     return {"loss_saliency": 0}

        vid_token_mask = targets["video_mask"].float()

        # Neg pair loss
        saliency_scores_neg = outputs["neg_saliency_scores"].clone()  # (N, L)
        # loss_neg_pair = torch.sigmoid(saliency_scores_neg).mean()
        
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * vid_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        if "saliency_label" in targets:
            saliency_contrast_label = targets["saliency_label"]
        else:
            saliency_contrast_label = targets["clip_mask"].float()

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        vid_token_mask = vid_token_mask.repeat([1, 2])
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        positive_count = 0
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                positive_count += 1
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        ## ablation 1
        loss_rank_contrastive = loss_rank_contrastive / self.rank_coef

        # ## ablation 2
        # loss_rank_contrastive = loss_rank_contrastive / positive_count

        if self.use_triplet:
            saliency_scores = outputs["saliency_scores"]  # (N, L)
            pos_indices = targets["pos_idx"]  # (N, #pairs)
            neg_indices = targets["neg_idx"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores), device=saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_triplet = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        # loss_saliency = loss_rank_contrastive
        # loss_saliency = loss_saliency + loss_rank_contrastive
        loss_saliency = loss_rank_contrastive + loss_neg_pair
        # loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        if self.use_triplet:
            loss_saliency = loss_saliency + loss_triplet
        return {"loss_saliency": loss_saliency}
    
    def loss_rec_ss(self, outputs, targets, indices, log=True):
        num_clips = targets["num_clips"]
        if self.multi_clip:
            moment_merge = []
            for m in targets["norm_moment"]:
                moment_merge.append(torch.stack([m['moments'].min(), m['moments'].max()]))
            moment_merge = torch.stack(moment_merge)
        else:
            moment_merge = targets["norm_moment"]
        for idx, moments in enumerate(torch.split(moment_merge, num_clips.tolist())):
            if idx == 0:
                iou_matrix = generalized_temporal_iou(moments, moments)
            else:
                iou_matrix = torch.block_diag(iou_matrix, generalized_temporal_iou(moments, moments))

        pos_mask = iou_matrix >= self.gamma
        tau = self.recss_tau

        clip_mask = targets["clip_mask"].unsqueeze(-1)
        clip_feat = outputs["projed_video_feat"] * clip_mask
        # clip_feat = outputs["enhanced_video_feat"] * clip_mask
        clip_feat = clip_feat.sum(dim=1) / clip_mask.sum(dim=1)
        
        # ## ablation 1
        # words_feat = outputs["projed_recon_feat"]

        # ## ablation 2
        # words_feat = outputs["recon_feat"]

        ## ablation3
        words_mask = outputs["expanded_words_mask"].unsqueeze(-1)
        words_feat = outputs["expanded_words_feat"] * words_mask
        words_feat = words_feat.sum(dim=1) / words_mask.sum(dim=1)

        ## ablation 1
        norm_clip_feat = F.normalize(clip_feat, dim=-1, p=2)
        norm_words_feat = F.normalize(words_feat, dim=-1, p=2)
        cos_sim = norm_clip_feat @ norm_words_feat.permute(1, 0)

        # ## ablation 2
        # cos_sim = clip_feat @ words_feat.permute(1, 0)

        # cos_sim = (norm_clip_feat * norm_words_feat).sum(dim=1)
        cos_sim = cos_sim / tau
        logits = cos_sim - torch.max(cos_sim, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-6)
        loss = - mean_log_prob_pos

        losses = {"loss_rec_ss": loss.mean()}
        return losses
    
    def loss_rec_fw(self, outputs, targets, indices, log=True):
        words_label = targets["words_label"]
        recfw_words_logit = outputs["recfw_words_logit"]
        words_mask = outputs["words_mask"]

        nll_loss, acc = self.cal_nll_loss(recfw_words_logit, words_label, words_mask)
        
        # if use_contrastive:
        #     neg_loss = self.loss_reconstruction(src_txt, outputs["recfw_neg_words_feat"])
        #     loss_contra = self.loss_contrastive_recon(pos_loss, neg_loss)
        #     losses["loss_recfw_contra"] = loss_contra.mean()

        return {"loss_rec_fw": nll_loss.mean(),
                "rec_fw_acc": acc}
    
    def cal_nll_loss(self, logit, idx, mask, weights=None):
        eps = 0.1
        acc = (logit.max(dim=-1)[1]==idx).float()
        mean_acc = (acc * mask).sum() / mask.sum()
        
        logit = logit.log_softmax(dim=-1)
        nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -logit.sum(dim=-1)
        nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
        if weights is None:
            nll_loss = nll_loss.masked_fill(mask == 0, 0)
            nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
        else:
            nll_loss = (nll_loss * weights).sum(dim=-1)

        return nll_loss.contiguous(), mean_acc

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "span": self.loss_spans,
            "label": self.loss_labels,
            "saliency": self.loss_saliency,
            "rec_ss": self.loss_rec_ss,
            "rec_fw": self.loss_rec_fw,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, is_training=True):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        # if self.use_matcher:
        indices = self.matcher(outputs_without_aux, targets)
        losses_target = self.losses
        # else:
        #     indices = None
        #     losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_target:
            if loss == "rec_fw" and not is_training:
                continue
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                # if self.use_matcher:
                indices = self.matcher(aux_outputs, targets)
                losses_target = self.losses
                # else:
                #     indices = None
                #     losses_target = ["saliency"]    
                for loss in losses_target:
                    if loss in ["saliency", "rec_ss", "rec_fw"]:
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)

        return losses, total_loss

