import pandas as pd
import numpy as np
import os
import pickle
import random
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, root, N = 20, S = 60):
        """
        INPUT:
            root: project root directory
            N: use first N pkt of a biflow
            S: use pkt in first S seconds of a biflow
        Features:
            'PL', 'DIR', 'WIN', 'IAT', 'TTL', 'FLG'
        """

        self.data_pth = os.path.join(root, 'data/CSE-CIC-IDS-2018/pktseq')

        self.H = N
        self.W = 6

        self.feature_all_np, self.label_str_all_list = self.get_pktseq(N, S)

        self.label_int_all_np, self.label_map = self.label_mapping(smpl_num_order=True, rand=False)
    
    def exp_init(self, base_cls: int, rand_tr_smpl=True, feature_minmax=None, feature_norm=None):
        self.data_tr, self.data_tst, self.target_tr, self.target_tst = self.tr_ts_split(random=rand_tr_smpl, base_cls=base_cls)

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
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Wednesday-14-02-2018.pkl",
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Thursday-15-02-2018.pkl",
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Friday-16-02-2018.pkl",
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Tuesday-20-02-2018.pkl",
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Wednesday-21-02-2018.pkl",
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Thursday-22-02-2018.pkl",
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Friday-23-02-2018.pkl",
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Wednesday-28-02-2018.pkl",
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Thursday-01-03-2018.pkl",
            "/data/home/liusm/data/CSE-CIC-IDS-2018/processed small pkl/Friday-02-03-2018.pkl"
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

        return x_np, y_list

    def label_mapping(self, smpl_num_order = True, rand = False):
        if smpl_num_order:
            label_map = {
                "BENIGN": 0,
                "DDoS-LOIC-HTTP": 1,
                "Bot": 2,
                "DoS-SlowHTTPTest": 3,
                "DoS-GoldenEye": 4,
                "DDoS-HOIC": 5,
                "DoS-Hulk": 6,
                "SSH-BruteForce": 7,
                "DoS-Slowloris": 8,
                "DDoS-LOIC-UDP": 9,
                "Brute Force-Web": 10,
                "Brute Force-XSS": 11,
                "SQL Injection": 12,
                "FTP-BruteForce": 13,
                "Infiltration": 14   
            }
        elif rand:
            big_cls = [
                "DDoS-LOIC-HTTP", "DDoS-HOIC", "Bot"
            ]
            
            small_cls = [
                "DoS-GoldenEye",
                "DoS-SlowHTTPTest",
                "DoS-Hulk",
                "SSH-BruteForce",
                "DoS-Slowloris",
                "DDoS-LOIC-UDP",
            ]
            label_map = {
                "BENIGN": 0,
                "Brute Force-Web":10,
                "Brute Force-XSS": 11,
                "SQL Injection": 12,
                "Infiltration": 13,
                "FTP-BruteForce": 14
            }
            random.shuffle(big_cls)
            random.shuffle(small_cls)
            for i, name in enumerate(big_cls):
                label_map[name] = i + 1
            for i, name in enumerate(small_cls):
                label_map[name] = i + 4
        else: 
            label_map = {
                "BENIGN": 0,
                "Bot": 1,
                "DDoS-LOIC-HTTP": 2,
                "DDoS-HOIC": 3,
                "FTP-BruteForce": 4,
                "DoS-Hulk": 5,
                "DoS-SlowHTTPTest": 6,
                "DoS-Slowloris": 7,
                "SSH-BruteForce": 8,
                "DoS-GoldenEye": 9,
                "DDoS-LOIC-UDP": 10,
                "Brute Force-Web": 11,
                "Brute Force-XSS": 12,
                "SQL Injection": 13,
                "Infiltration": 14   
            }

        label_int_list = [label_map[label] for label in self.label_str_all_list]
        label_int_np = np.array(label_int_list, dtype=np.int64)

        return label_int_np, label_map


    def tr_ts_split(self, random = True, base_cls = 5):
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
            elif self.label_map[label] < base_cls:
                n_train = 2500
            elif label != "Infiltration":
                n_train = 5
            else:
                n_train = 4

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
