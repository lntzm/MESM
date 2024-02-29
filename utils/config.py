import os
import time
import json
import torch
import shutil
import argparse

from .func_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown

# gwork, share_MLP, num_recfw_layers, rec_fw_contra,
# rec_ss, num_recss_layers, recfw_margin
# loss_svd_coef, loss_recss_coef, recss_tau, visualization

class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()

        parser.add_argument("--config_file", type=str, default=None)
        
        ## dataset
        parser.add_argument("--dataset_name", type=str,
                            choices=['charades', 'charades-cg', 'charades-cd', 'qvhighlights', 'tacos'])
        parser.add_argument("--ann_path", type=str)
        parser.add_argument("--feat_files", type=str, nargs="+",
                            help="video feature dirs. If more than one, will concat their features. "
                                 "Note that sub ctx features are also accepted here.")
        parser.add_argument("--use_tef", default=False, action="store_true")
        parser.add_argument("--clip_len", type=int, default=1)
        parser.add_argument("--max_words_l", type=int, default=32)
        parser.add_argument("--max_video_l", type=int, default=75)
        parser.add_argument("--tokenizer_type", type=str, choices=['CLIP', 'GloVeSimple', 'GloVeNLTK'], default='CLIP')
        parser.add_argument("--load_vocab_pkl", default=False, action="store_true",
                            help="Only for tokenizer_type==GloveNLTK, fasttrack")
        parser.add_argument("--bpe_path", type=str, default="data/bpe_simple_vocab_16e6.txt.gz")
        parser.add_argument("--normalize_video", action="store_true")
        parser.add_argument("--normalize_txt", action="store_true")
        parser.add_argument("--contra_samples", type=int, default=2)
        parser.add_argument("--batch_size", type=int, default=12)
        parser.add_argument("--eval_batch_size", type=int, default=-1)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pin_memory", action="store_true")
        parser.add_argument("--vocab_size", type=int, default=1111)
        parser.add_argument("--max_windows", type=int, default=5)
        parser.add_argument("--max_gather_size", type=int, default=-1)

        ## model
        parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
        # TextEncoder
        parser.add_argument("--text_model_path", type=str, default='data/clip_text_encoder.pth')
        # T2VEncoder & Transformer
        parser.add_argument("--share_MLP", default=False, action="store_true")
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--nheads", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=1024)
        parser.add_argument("--num_recfw_layers", type=int, default=2)
        parser.add_argument("--t2v_layers", type=int, default=2)
        parser.add_argument("--enc_layers", type=int, default=2)
        parser.add_argument("--dec_layers", type=int, default=2)
        parser.add_argument("--pre_norm", action="store_true")
        # Position Embedding
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # DETR
        parser.add_argument('--input_dropout', default=0.5, type=float,
                            help="Dropout applied in input")
        parser.add_argument("--v_feat_dim", type=int, help="video feature dim")
        parser.add_argument("--t_feat_dim", type=int, help="text/query feature dim")
        parser.add_argument('--num_queries', default=10, type=int,
                            help="Number of query slots")
        parser.add_argument("--use_txt_pos", action="store_true",
                            help="use position_embedding for text as well.")
        parser.add_argument("--n_input_proj", type=int, default=2,
                            help="#layers to encoder input")
        parser.add_argument("--rec_fw", default=False, action="store_true", help="frame-word level MESM")
        parser.add_argument("--rec_ss", default=False, action="store_true", help="segment-sentence level MESM")
        parser.add_argument("--num_recss_layers", type=int, default=4)

        ## matcher
        parser.add_argument('--set_cost_span', default=10, type=float,
                            help="L1 span coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=1, type=float,
                            help="giou span coefficient in the matching cost")
        parser.add_argument('--set_cost_class', default=4, type=float,
                            help="Class coefficient in the matching cost")

        ## criterion
        parser.add_argument("--span_loss_type", type=str, default="l1", choices=["l1", "ce"])
        parser.add_argument("--aux_loss", default=False, action="store_true",
                            help="Auxiliary decoding losses (loss at each layer)")
        parser.add_argument("--rank_coef", type=float, default=12.0)
        parser.add_argument("--use_triplet", default=False, action="store_true")
        parser.add_argument("--saliency_margin", type=float, default=0.2)
        parser.add_argument("--loss_span_coef", default=10, type=float)
        parser.add_argument("--loss_giou_coef", default=1, type=float)
        parser.add_argument("--loss_label_coef", default=4, type=float)
        parser.add_argument("--loss_saliency_coef", default=1, type=float)
        parser.add_argument("--eos_coef", default=0.1, type=float,
                            help="Relative classification weight of the no-object class")
        parser.add_argument("--loss_recfw_coef", default=0, type=float)
        parser.add_argument("--loss_recss_coef", default=0, type=float)
        parser.add_argument("--iou_gamma", default=0.9, type=float)
        parser.add_argument("--recss_tau", default=0.5, type=float)

        ## train
        parser.add_argument("--exp_id", type=str, default=None,
                            help="id of this run, required at training")
        parser.add_argument("--seed", type=int, default=2019)
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="learning rate")
        parser.add_argument("--lr_drop", type=int, default=400,
                            help="drop learning rate to 1/10 every lr_drop epochs")
        parser.add_argument("--gamma", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=1e-4,
                            help="weight decay")
        parser.add_argument("--n_epoch", type=int, default=200,
                            help="number of epochs to run")
        parser.add_argument("--grad_clip", type=float, default=0.1,
                            help="perform gradient clip, -1: disable")
        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        parser.add_argument("--start_epoch", type=int, default=None,
                            help="if None, will be set automatically when using --resume_all")
        parser.add_argument("--eval_untrained", action="store_true",
                            help="Evaluate on un-trained model")
        parser.add_argument("--max_es_cnt", type=int, default=200,
                            help="number of epochs to early stop, use -1 to disable early stop")
        parser.add_argument("--save_interval", type=int, default=50)
        parser.add_argument("--result_root", type=str, default='./results')
        parser.add_argument("--ctx_mode", type=str, default=None)
        parser.add_argument("--stop_score", type=str, default='mAP')
        
        ## eval
        parser.add_argument("--eval_epoch_interval", type=int, default=1)
        parser.add_argument("--sort_results", action="store_true",
                            help="sort results, not use this for moment query visualization")
        parser.add_argument("--nms_thd", type=float, default=-1,
                            help="additionally use non-maximum suppression "
                                 "(or non-minimum suppression for distance)"
                                 "to post-processing the predictions. "
                                 "-1: do not use nms. [0, 1]")
        parser.add_argument("--max_ts_val", type=float, default=150)
        parser.add_argument("--max_before_nms", type=int, default=10)
        parser.add_argument("--max_after_nms", type=int, default=10)

        self.parser = parser

    def load_config(self, opt):
        opt.__dict__.update(load_json(opt.config_file))

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        # if not isinstance(self, TestOptions):
        option_file_path = os.path.join(opt.result_dir, self.saved_option_filename)  # not yaml file indeed
        save_json(args, option_file_path, save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        if opt.config_file:
            self.load_config(opt)

        if isinstance(self, TestOptions):
            opt.is_inference = True
            saved_options = load_json(os.path.join(opt.trained_result_dir, self.saved_option_filename))
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["config_file", "num_workers", "nms_thd", "device",
                               "resume_all", "sort_results", "max_ts_val",
                               "ann_path", "is_inference",
                               "feat_files", "bpe_path", "text_model_path"]:
                    setattr(opt, arg, saved_options[arg])
            if opt.trained_result_dir is None:
                assert opt.resume is not None
                opt.trained_result_dir = os.path.dirname(opt.resume)
            else:
                if opt.dataset_name == "qvhighlights":
                    split_name = "val"
                else:
                    split_name = "test"
                opt.resume = os.path.join(opt.trained_result_dir, f"model_{split_name}_best.ckpt")

            if opt.inference_result_dir is not None:
                opt.result_root = opt.inference_result_dir

            save_name = "-".join([opt.dataset_name, "eval", opt.inference_id, time.strftime("%Y_%m_%d_%H_%M_%S")])
            opt.result_dir = os.path.join(opt.result_root, save_name)
            mkdirp(opt.result_dir)
        else:
            opt.is_inference = False
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")
            
            if opt.eval_batch_size == -1:
                opt.eval_batch_size = opt.batch_size

            ctx_str = opt.ctx_mode + "sub" if any(["sub_ctx" in p for p in opt.feat_files]) else opt.ctx_mode
            if ctx_str is None:
                save_name = "-".join([opt.dataset_name, opt.exp_id, time.strftime("%Y_%m_%d_%H_%M_%S")])
            else:
                save_name = "-".join([opt.dataset_name, ctx_str, opt.exp_id, time.strftime("%Y_%m_%d_%H_%M_%S")])
            opt.result_dir = os.path.join(opt.result_root, save_name)
            mkdirp(opt.result_dir)
            save_fns = ['model/model.py', 'model/transformer.py', 'model/criterion.py']
            for save_fn in save_fns:
                shutil.copyfile(save_fn, os.path.join(opt.result_dir, os.path.basename(save_fn)))

            # # save a copy of current code
            # code_dir = os.path.dirname(os.path.realpath(__file__))
            # code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            # make_zipfile(code_dir, code_zip_filename,
            #              enclosing_dir="code",
            #              exclude_dirs_substring="results",
            #              exclude_dirs=["results", "debug_results", "__pycache__"],
            #              exclude_extensions=[".pyc", ".ipynb", ".swap"], )

        self.display_save(opt)

        opt.ckpt_filepath = os.path.join(opt.result_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.result_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.result_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.result_dir, self.tensorboard_log_dir)
        opt.device = torch.device(opt.device)

        if opt.use_tef:
            opt.v_feat_dim += 2

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""

    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--inference_id", type=str, help="evaluation id", default="")
        self.parser.add_argument("--inference_result_dir", type=str, default=None,
                                 help="dir to save results, if not set, fall back to training results_dir")
        self.parser.add_argument("--trained_result_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")

