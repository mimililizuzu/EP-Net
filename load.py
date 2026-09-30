import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DataScheduler:
    def __init__(self, args, dataset, tr_trans, ts_trans):
        self.args = args
        self.dataset = dataset
        self.tr_trans = tr_trans
        self.ts_trans = ts_trans
        self.cls_encountered = []
        self.cls_remain = list(range(args.tol_cls))

    def get_dataloader(self, session: int, ses_cls_list: list):
        self.cls_encountered.extend(ses_cls_list)
        self.cls_remain = [cls for cls in self.cls_remain if cls not in ses_cls_list]
        if session == 0:    
            train_x, train_y = self.dataset.select_train_cls_smpl(
                cls_list = ses_cls_list, 
                sampling = self.args.sampling,
                k = self.args.base_K, 
                bn_clusters = self.args.bn_cluster,
                random = self.args.base_rand_tr_sampling
                )
            test_x, test_y = self.dataset.select_test_cls_smpl(
                cls_list = self.cls_encountered, 
                k = self.args.test_K, 
                random = self.args.rand_ts_sampling
                )
            
            trainset = LoaderDataset(train_x, train_y, transform=self.tr_trans)
            testset = LoaderDataset(test_x, test_y, transform=self.ts_trans)
            
            trainloader = DataLoader(
                dataset=trainset,
                batch_size=self.args.base_train_batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=False
            )
            testloader = DataLoader(
                dataset=testset,
                batch_size=self.args.test_batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=False
            )
        else:   
            train_x, train_y = self.dataset.select_train_cls_smpl(
                cls_list = ses_cls_list, 
                sampling = self.args.sampling,
                k = self.args.K, 
                random = self.args.new_rand_tr_sampling
                )
            test_x, test_y = self.dataset.select_test_cls_smpl(
                cls_list = self.cls_encountered, 
                k = self.args.test_K, 
                random = self.args.rand_ts_sampling
                )
            
            trainset = LoaderDataset(train_x, train_y, transform=self.ts_trans)
            testset = LoaderDataset(test_x, test_y, transform=self.ts_trans)

            trainloader = DataLoader(
                dataset=trainset,
                batch_size=self.args.K if self.args.new_train_batch_size == 0 else self.args.new_train_batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=True
            )
            testloader = DataLoader(
                dataset=testset,
                batch_size= self.args.test_batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True
            )

        return trainloader, testloader


class LoaderDataset(Dataset):
    def __init__(self, data_np, target_np, transform):
        self.data = data_np
        self.target = target_np
        self.transform = transform
    
    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        if self.transform is not None:
            data = self.transform(torch.tensor(data, dtype=torch.float32))
        return data, target
    
    def __len__(self):
        return len(self.data)
