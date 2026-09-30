import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn

def setseeds(seed):
    if seed is None:
        torch.backends.cudnn.benchmark = True
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setgpu(gpuseq: str):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuseq
    return len([int(n) for n in gpuseq.split(',')])

def save_log(pth, list):
    f = open(pth, mode='w')
    for line in list:
        f.write(str(line)+'\n')
    f.close()

def cls_acc(conf_mat):
    cls_acc = []
    for c in range(conf_mat.shape[0]):
        tp = conf_mat[c, c]
        tol_smpl = conf_mat[c, :].sum()
        if tol_smpl == 0:
            print(f"WARNING: cls {c} has no test sample!")
            acc = 1.0
        else:
            acc = tp/tol_smpl
        cls_acc.append(acc)
    return cls_acc

def cal_log_avg(args, rep_log):
    ses_num = args.session

    keys = [
        'acc', 'avg_cls_acc', 'recall', 'precision', 'F1', 'conf_mat',
        'base_acc', 'new_acc', 'base_avg_cls_acc', 'new_avg_cls_acc',
        'speed'
    ]
    avglog = {k:[] for k in keys}

    for s in range(ses_num):
        ses_vals = {k:[] for k in keys}

        for res in rep_log:
            for k in keys:
                if k != 'conf_mat':
                    val = res[k][s]
                    if hasattr(val, 'item'):
                        val = val.item()
                else:
                    val = res[k][s]
                    if hasattr(val, 'cpu'):
                        val = val.cpu().numpy()
                    val = np.round(val, decimals=5)
                    val = cm_num_to_prob(np.array(val))
                ses_vals[k].append(val)

        for k in keys:
            if k != 'conf_mat':
                avglog[k].append(float(np.mean(ses_vals[k])))
            else:
                avglog[k].append(np.mean(ses_vals[k], axis=0))

    return avglog


def cm_num_to_prob(cm_num, demical = 5):
    cm_prob = np.zeros(cm_num.shape)
    for i in range(cm_num.shape[0]):
        cm_prob[i] = cm_num[i]/np.sum(cm_num[i])
    cm_prob = np.round(cm_prob, decimals = demical)
    return cm_prob

def save_cm_img(cm_np, o_dir, label_map):
    cm_np = np.nan_to_num(cm_np, nan=0.0)
    id2str = {v: k.replace('\x96', '-') for k, v in label_map.items()}
    classes = [id2str[i] for i in range(cm_np.shape[0])]

    n = len(classes)
    plt.figure(figsize=(max(10, int(n * 0.5)), max(8, int(n * 0.5))))
    seaborn.heatmap(
        cm_np,
        annot=True,
        fmt='.2f',
        cmap='jet',
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.2,
        linecolor='white',
        square=True,
        cbar=False
    )
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()

    os.makedirs(o_dir, exist_ok=True)
    plt.savefig(os.path.join(o_dir, 'confusion_matrix.jpg'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(o_dir, 'confusion_matrix.pdf'), bbox_inches='tight', format='pdf')
    plt.close()
