import pandas as pd
import numpy as np
import os
import pickle
import random
from sklearn.model_selection import train_test_split
from pathlib import Path

class Dataset:
    def __init__(self, root, N = 20, S = 60):
        """
        INPUT:
            root: project root directory
            N: use first N pkt of a biflow
            S: use pkt in first S seconds of a biflow
        ATTRIBUTES:
            feature_all_np: all samples feature numpy (N_BIFLOW, N, 6)
            label_str_all_list: all samples str labels (N_BIFLOW, )
            label_int_all_np: all samples int labels (N_BIFLOW, )
            label_map: str label -> int label dict
            H: height of 2d vector of a biflow (pkt number)
            W: width of 2d vector of a biflow (features of a pkt)
        Features:
            'PL', 'DIR', 'WIN', 'IAT', 'TTL', 'FLG'
        """

        self.data_pth = os.path.join(root, 'data/CICIDS2017/pktseq')

        self.H = N
        self.W = 6

        self.feature_all_np, self.label_str_all_list = self.get_pktseq(N, S)

        self.label_int_all_np, self.label_map = self.label_mapping(smpl_num_order=True, rand=False)

    def exp_init(self, base_cls: int, rand_tr_smpl=True, feature_minmax=None, feature_norm=None):
        self.data_tr, self.data_tst, self.target_tr, self.target_tst = self.tr_ts_split(random=rand_tr_smpl)

        self.feature_minmax = feature_minmax
        if feature_minmax is not None:
            self.min_max_norm(base_cls)

        self.feature_norm = feature_norm
        if feature_norm is not None:
            self.mean, self.std = self.get_mean_std(base_cls)
        else:
            self.mean, self.std = None, None


    def select_train_cls_smpl(self, cls_list:list, sampling:str='none', k:int=None, bn_clusters:int=1, random:bool=True):
        cls_n_list = {}
        cls_idx_list = {}
        for i in cls_list:
            cls_smpl_idx = np.where(i == self.target_tr)[0]
            cls_idx_list[i] = cls_smpl_idx
            cls_n_list[i] = len(cls_smpl_idx)

        data = []
        target = []
        if sampling == 'none':
            for idx in cls_idx_list.values():
                data.append(self.data_tr[idx])
                target.append(self.target_tr[idx])
        else:
            raise ValueError(f"Unsupported sampling method: {sampling}. Supported: ['none']")
        data = np.vstack(data)
        target = np.hstack(target)

        return data, target
    
    def select_test_cls_smpl(self, cls_list:list, k:int = None, random:bool = True):
        data = []
        target = []
        for i in cls_list:
            cls_smpl_idx = np.where(i == self.target_tst)[0]
            if k is not None:
                if random and k < len(cls_smpl_idx):
                    cls_smpl_idx = np.random.choice(cls_smpl_idx, k, replace=False)
                else:
                    cls_smpl_idx = cls_smpl_idx[:k]
            data.append(self.data_tst[cls_smpl_idx])
            target.append(self.target_tst[cls_smpl_idx])
        data = np.vstack(data)
        target = np.hstack(target)
        return data, target

    def get_pktseq(self, N = 20, S = 60):
        pkl_list = [
            ".../Monday-WorkingHours.pkl",
            ".../Tuesday-WorkingHours.pkl",
            ".../Wednesday-WorkingHours.pkl",
            ".../Thursday-WorkingHours.pkl",
            ".../Friday-WorkingHours.pkl",
        ]

        x_list = []
        y_list = []
        
        print("loading pkls...")
        for pkl in pkl_list:
            if not os.path.exists(pkl):
                raise FileNotFoundError(f"File {pkl} not found.")
            with open(pkl, 'rb') as f:
                bf_dict = pickle.load(f)

            for (bf_id, label), pkts in bf_dict.items():
                t0 = float(pkts[0]["ts"])
                pkts_intime = [p for p in pkts if float(p["ts"]) - t0 <= float(S)]

                vec = np.zeros((N, self.W), dtype=np.float32)
                for row_id, pkt_dict in enumerate(pkts_intime[:N]):
                    vec[row_id, 0] = float(pkt_dict["PL"])
                    vec[row_id, 1] = float(pkt_dict["DIR"])
                    vec[row_id, 2] = float(pkt_dict["WIN"])
                    vec[row_id, 3] = float(pkt_dict["IAT"])
                    vec[row_id, 4] = float(pkt_dict["TTL"])
                    vec[row_id, 5] = float(pkt_dict["FLG"])

                x_list.append(vec)
                y_list.append(label)
            
        x_np = np.stack(x_list, axis=0)

        hb_id = y_list.index("Heartbleed")
        dup_x = x_np[hb_id].copy()
        x_np = np.vstack([x_np, dup_x[None, ...]])
        y_list.append("Heartbleed")

        return x_np, y_list

    def label_mapping(self, smpl_num_order = True, rand = False):
        if smpl_num_order:
            label_map = {
                "BENIGN": 0,
                "PortScan": 1,
                "DDoS": 2,
                "DoS Hulk": 3,
                "DoS GoldenEye": 4,
                "DoS Slowhttptest": 5,
                "FTP-Patator": 6,
                "DoS slowloris": 7,
                "SSH-Patator": 8,
                "Web Attack \x96 Brute Force": 9,
                "Bot": 10,
                "Web Attack \x96 XSS": 11,
                "Web Attack \x96 Sql Injection": 12,
                "Infiltration": 13,
                "Heartbleed": 14
            }
        elif rand:
            big_cls = [
                "PortScan", "DDoS", "DoS Hulk", "DoS GoldenEye",
                "DoS Slowhttptest", "FTP-Patator", "DoS slowloris", "SSH-Patator"
            ]
            small_cls = [
                "Web Attack \x96 Brute Force", "Bot"
            ]
            label_map = {
                "BENIGN": 0,
                "Web Attack \x96 XSS": 11,
                "Web Attack \x96 Sql Injection": 12,
                "Infiltration": 13,
                "Heartbleed": 14
            }
            random.shuffle(big_cls)
            random.shuffle(small_cls)
            for i, name in enumerate(big_cls):
                label_map[name] = i + 1
            for i, name in enumerate(small_cls):
                label_map[name] = i + 9
        else: 
            label_map = {
                "BENIGN": 0,
                "PortScan": 5,
                "DDoS": 6,
                "DoS Hulk": 1,
                "DoS GoldenEye": 2,
                "DoS Slowhttptest": 3,
                "FTP-Patator": 7,
                "DoS slowloris": 4,
                "SSH-Patator": 8,
                "Web Attack \x96 Brute Force": 9,
                "Bot": 10,
                "Web Attack \x96 XSS": 11,
                "Web Attack \x96 Sql Injection": 12,
                "Infiltration": 13,
                "Heartbleed": 14
            }

        label_int_list = [label_map[label] for label in self.label_str_all_list]
        label_int_np = np.array(label_int_list, dtype=np.int64)

        return label_int_np, label_map


    def tr_ts_split(self, random = True):
        if random is None:
            data_tr, data_tst, target_tr, target_tst = train_test_split(
                self.feature_all_np,
                self.label_int_all_np,
                test_size=0.2,
                random_state=None,
                shuffle=True,
                stratify=self.label_int_all_np
            )
            return data_tr, data_tst, target_tr, target_tst
        
        data_tr, data_tst, target_tr, target_tst = [], [], [], []
        for label in self.label_map:
            if label == "BENIGN":
                n_train = 10000
            elif label == "PortScan":
                n_train = 2500
            elif label == "DDoS":
                n_train = 2500
            elif label == "DoS Hulk": 
                n_train = 2500
            elif label == "DoS GoldenEye":  
                n_train = 2500
            elif label == "DoS Slowhttptest":
                n_train = 5
            elif label == "FTP-Patator": 
                n_train = 5
            elif label == "DoS slowloris":
                n_train = 5
            elif label == "SSH-Patator":
                n_train = 5
            elif label == "Web Attack \x96 Brute Force":    
                n_train = 5
            elif label == "Bot":
                n_train = 5
            elif label == "Web Attack \x96 XSS":
                n_train = 5
            elif label == "Web Attack \x96 Sql Injection":
                n_train = 5
            elif label == "Infiltration":
                n_train = 5
            elif label == "Heartbleed":
                n_train = 1

            cls_idx = np.where(self.label_int_all_np == self.label_map[label])[0]
            cls_data = self.feature_all_np[cls_idx]
            cls_target = self.label_int_all_np[cls_idx]
            if random:
                tr_idx = np.random.choice(len(cls_data), n_train, replace=False)
                tst_idx = list(set(range(len(cls_data)))-set(tr_idx))
            else:
                tr_idx = list(range(n_train))
                tst_idx = list(range(len(cls_data)))[n_train:]
            data_tr_c, target_tr_c, data_tst_c, target_tst_c = cls_data[tr_idx], cls_target[tr_idx], cls_data[tst_idx], cls_target[tst_idx]
            data_tr.append(data_tr_c)
            target_tr.append(target_tr_c)
            data_tst.append(data_tst_c)
            target_tst.append(target_tst_c)
        data_tr = np.vstack(data_tr)
        data_tst = np.vstack(data_tst)
        target_tr = np.hstack(target_tr)
        target_tst = np.hstack(target_tst)
        
        return data_tr, data_tst, target_tr, target_tst

    def min_max_norm(self, base_cls):
        base_mask = self.target_tr < base_cls
        base_tr = self.data_tr[base_mask]
        if self.feature_minmax:
            min_ = np.min(base_tr, axis=(0,1))
            max_ = np.max(base_tr, axis=(0,1))
        else:
            min_ = np.min(base_tr)
            max_ = np.max(base_tr)
        norm_range = max_ - min_
        self.data_tr = (self.data_tr - min_)/norm_range
        self.data_tst = (self.data_tst - min_)/norm_range

    def get_mean_std(self, base_cls):
        base_mask = self.target_tr < base_cls
        base_tr = self.data_tr[base_mask]
        if self.feature_norm:
            mean_ = base_tr.mean(axis=(0, 1)) 
            std_ = base_tr.std(axis=(0, 1), ddof=0)
        else:
            mean_ = base_tr.mean()
            std_ = base_tr.std(ddof=0)
        return mean_, std_
