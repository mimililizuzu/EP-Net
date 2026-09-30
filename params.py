import argparse
import os
import math
from datetime import datetime

parser = argparse.ArgumentParser(description='EP-Net: Few-Shot Class-Incremental Learning for Encrypted Network Intrusion Detection')
parser.add_argument('--dataset', type=str, default='cicids2017', choices=['cicids2017', 'csecicids2018'],
                    help='dataset used for experiment')
parser.add_argument('--root', type=str, default=None, help='root path of this project')
parser.add_argument('--o_pth', type=str, default=None, help='output path')
parser.add_argument('--gpuseq', type=str, default='0', help='str sequence of gpus used')
parser.add_argument('--seed', type=int, default=1, help='init seed random number')
parser.add_argument('--repeat_n', type=int, default=10, help='number of repeated experiments')

args_ = parser.parse_args()

# ---------- fixed hyper-parameters of the paper (Table 4) ----------
# traffic representation: bi-flow -> P x 6 matrix -> 32x32 grayscale image
args_.Np = 20                     # P: collected packet number of a bi-flow
args_.S = 60                      # S: maximum bi-flow packet collection time in seconds
args_.feature_minmax = None
args_.feature_norm = True         # column-wise Z-score normalization
args_.rand_tr_tst_split = True

# FSCIL protocol
args_.tol_cls = 11 if args_.dataset == 'cicids2017' else 13
args_.base_cls = 5                # benign class + 4 base attack classes
args_.N = 2                       # new attack classes per incremental task
args_.K = 5                       # k: labeled samples of each new class
args_.session = math.ceil((args_.tol_cls - args_.base_cls) / args_.N) + 1
args_.sampling = 'none'

# benign class division
args_.bn_cluster = 4              # Cbn: pseudo label number of benign class

# base task training
args_.base_epochs = 50
args_.lr_init = 0.1
args_.lr_milestone = [0.8, 0.9]   # lr decays after 40 and 45 epochs
args_.lr_dec = 0.1
args_.weight_dec = 5e-4
args_.momentum = 0.9              # SGD with Nesterov Accelerated Gradient
args_.base_train_batch_size = 128
args_.base_K = 2500               # training samples of each base attack class (benign: 10000)
args_.base_rand_tr_sampling = True

# data augmentation: P/2 consecutive rows and 5 columns of the matrix
args_.num_aug = 2                 # 2 extra augmented samples (3 views in total)
args_.aug_row = 10                # P/2
args_.aug_col = 5

# loss: L = Lce + lambda_nce * Lnce + lambda_cp * Lcp
args_.ce_tau = 32.0               # 1/tau in Lce
args_.ssc_tau = 32.0              # 1/tau in Lnce
args_.ssc_lamb = 0.1              # lambda_nce
args_.inter_lamb = 0.2 if args_.dataset == 'cicids2017' else 0.6  # lambda_cp

# model: ResNet20 with 128-d feature embedding
args_.backbone = 'ResNet20'
args_.embd_dim = 128
args_.fw_tau = 16.0               # logit scale at test time

# extra-prototypes selection (inner-class K-Means clustering)
args_.extra_proto = 5             # Ep: extra-prototypes of each class

# incremental updating and testing
args_.new_rand_tr_sampling = True
args_.new_train_batch_size = 0
args_.test_K = 1000               # test samples of each class
args_.rand_ts_sampling = True
args_.test_batch_size = 100

def get_args():
    return args_

def set_rootpth(args):
    if args.root is None:
        args.root = os.path.dirname(os.path.abspath(__file__))

def set_outpth(args):
    if args.o_pth is None:
        args.o_pth = f"./logs/{args.dataset}/{args.backbone}/{args.session}sess/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
        os.makedirs(args.o_pth, exist_ok=True)
