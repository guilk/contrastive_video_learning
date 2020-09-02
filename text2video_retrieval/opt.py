import os, sys
import argparse
ROOT = './'
VFEAT_DIR = os.path.join(ROOT, 'vfeat/')
TFEAT_DIR = os.path.join(ROOT, 'vfeat/')
CAPTION_FILE = os.path.join(ROOT,'vtt_caps.token')
SPLIT_DIR = './splits'
num_caps=1

def parse_arguments():
    parser = argparse.ArgumentParser(description='Bernie')

    # I/O 
    parser.add_argument("--tfeat_dir", type=str, default=TFEAT_DIR, help='textual feat dir')
    parser.add_argument("--vfeat_dir", type=str, default=VFEAT_DIR, help='visual feat dir')
    parser.add_argument("--src_cap", type=str, default=CAPTION_FILE, help="cap file")
    parser.add_argument("--tgt_cap", type=str, default=CAPTION_FILE, help="cap file")
    parser.add_argument("--split_dir", type=str, default=SPLIT_DIR, help='split file')
    parser.add_argument("--model_dir", type=str, default='model', help='model dir')
    parser.add_argument("--model_name", type=str, default="baseline", help='model name')
    parser.add_argument("--model_ckpt", type=str, default="", help='model ckpt name to resume train/test')
    parser.add_argument("--model_ckpt_name", type=str, default="", help='model ckpt name')
    
    # data config
    parser.add_argument("--num_roi", type=int, default=128, help='number of roi')
    parser.add_argument("--vfeat_dim", type=int, default=2048, help='visual feat dim')
    parser.add_argument("--vfeat_type", type=str, default='resnet', help='the type of visual feature')
    parser.add_argument("--tfeat_dim", type=int, default=300, help='visual feat dim')

    # inference
    parser.add_argument("--is_inf", action='store_true', help='inference the model, no eval')
    parser.add_argument("--shared_bn", action='store_true', help='inference the model, no eval')
    
    # model configs
    parser.add_argument("--mean", action='store_true', help='no attn just mean')
    parser.add_argument("--hidden_dim", type=int, default=1024, help='hidden dim')
    parser.add_argument("--emb_dim", type=int, default=300, help='emb_dim')
    parser.add_argument("--margin", type=float, default=0.2, help="margin in the rankning loss function (alpha-S(I,T)+S(I,T')" )
    parser.add_argument("--glove", action='store_true', help='use glove')

    # training configs
    parser.add_argument("--is_train", action='store_true', help='train the model')
    parser.add_argument("--batch_size", type=int, default=128, help='batch_size')
    parser.add_argument("--num_epochs", type=int, default=20, help='number of epochs')
    parser.add_argument("--validate_every", type=int, default=1, help='validate every x epochs')
    parser.add_argument("--load_best", action='store_true', help='load the best model')

    parser.add_argument("--clip", type=float, default=2.0, help='gradient clip')
    parser.add_argument("--wd", type=float, default=0.000000, help='weight decay')

    parser.add_argument("--optimizer", type=str, default='Adam', help='optimizer')
    parser.add_argument("--lr", type=float, default=0.0001, help='initial learning rate')
    parser.add_argument("--step_size", type=int, default=16, help='shrink lr every lr_step')
    parser.add_argument("--gamma", type=float, default=10, help='shrink lr to 1/x')
    
    parser.add_argument("--text_encoder", default='rnn', choices=['rnn', 'cnn', 'transformer', 'linear'], help='text encoder type')
    parser.add_argument("--vis_encoder",  default='rnn', choices=['rnn', 'cnn', 'transformer', 'linear'], help='vis encoder type')
    
    parser.add_argument("--cnn2", action='store_true', help='l2norm text encoding')
    parser.add_argument("--vin_norm", action='store_true', help='vin norm')

    parser.add_argument("--text_norm", action='store_true', help='l2norm text encoding')
    parser.add_argument("--vis_norm", action='store_true', help='l2norm visual encoding')
    parser.add_argument("--factor", type=int, default=1, help='factor*hidden for evaluator')
    parser.add_argument("--www", type=float, default=0, help='www')
    parser.add_argument("--dp", type=float, default=0.3, help='dropout rate')
    parser.add_argument("--bn", action='store_true', help='use bn')

    parser.add_argument("--step", type=int, default=3, help='heads')
    config = parser.parse_args()
    if config.vfeat_type == 'resnet' or config.vfeat_type == 'resnext':
        config.vfeat_dim = 2048
    elif config.vfeat_type == 'efficientnet':
        config.vfeat_dim = 2560

    if not os.path.exists(os.path.join(config.model_dir, config.model_name)):
        os.makedirs(os.path.join(config.model_dir, config.model_name))
    config.model_path = os.path.join(config.model_dir, config.model_name)
    return config
