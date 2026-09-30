import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, args, block, layers, feat_dim=32, mean=None, std=None):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)

        self.flat_dim = 64
        self.feat_dim = feat_dim
        if self.flat_dim != self.feat_dim:
            self.fc = nn.Linear(self.flat_dim, self.feat_dim)

        self.transform = Trans(mean, std)
        if args.aug_row is None or args.aug_col is None:
            self.aug_transform = AugTrans(args.num_aug, mean, std)
        else:
            self.aug_transform = AugTrans2(args.num_aug, mean, std, args.aug_row, args.aug_col)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.flat_dim != self.feat_dim:
            x = self.fc(x)

        return x
    

class Trans:
    def __init__(self, mean, std):
        if mean is None and std is None:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.unsqueeze(0)),
                transforms.Resize((32,32)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        elif mean.ndim == 0 and std.ndim == 0:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.unsqueeze(0)),
                transforms.Normalize((mean,), (std,)),
                transforms.Resize((32,32)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        elif mean.ndim == 1 and std.ndim == 1:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.t().unsqueeze(-1)),
                transforms.Normalize(mean, std),
                transforms.Lambda(lambda x: x.permute(2, 1, 0)),
                transforms.Resize((32,32)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        else:
            raise ValueError(f"mean and std are unacceptable.")

    def __call__(self, x):
        return self.transform(x)

class AugTrans:
    def __init__(self, num_aug, mean, std):
        self.num_aug = num_aug
        if mean is None and std is None:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.unsqueeze(0)),
                transforms.Resize((32,32)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
            self.aug_transform = transforms.Compose([
                transforms.Lambda(lambda x: x.unsqueeze(0)),
                transforms.Resize((32,32)),
                transforms.RandomChoice([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomResizedCrop(32, scale=(0.25, 1.0), ratio=(1, 1))
                ]),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        elif mean.ndim == 0 and std.ndim == 0:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.unsqueeze(0)),
                transforms.Normalize((mean,), (std,)),
                transforms.Resize((32,32)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
            self.aug_transform = transforms.Compose([
                transforms.Lambda(lambda x: x.unsqueeze(0)),
                transforms.Normalize((mean,), (std,)),
                transforms.Resize((32,32)),
                transforms.RandomChoice([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomResizedCrop(32, scale=(0.25, 1.0), ratio=(1, 1))
                ]),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        elif mean.ndim == 1 and std.ndim == 1:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.t().unsqueeze(-1)),
                transforms.Normalize(mean, std),
                transforms.Lambda(lambda x: x.permute(2, 1, 0)),
                transforms.Resize((32,32)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
            self.aug_transform = transforms.Compose([
                transforms.Lambda(lambda x: x.t().unsqueeze(-1)),
                transforms.Normalize(mean, std),
                transforms.Lambda(lambda x: x.permute(2, 1, 0)),
                transforms.Resize((32,32)),
                transforms.RandomChoice([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomResizedCrop(32, scale=(0.25, 1.0), ratio=(1, 1))
                ]),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        else:
            raise ValueError(f"mean and std are unacceptable.")
    
    def __call__(self, x):
        if self.num_aug == 0:
            return [self.transform(x)]
        y = []
        for _ in range(self.num_aug+1):
            y.append(self.aug_transform(x))
        return y

class AugTrans2:
    def __init__(self, num_aug, mean, std, aug_row, aug_col):
        self.num_aug = num_aug
        self.aug_row = aug_row
        self.aug_col = aug_col
        if mean is None and std is None:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.unsqueeze(0)),
                transforms.Resize((32,32)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        elif mean.ndim == 0 and std.ndim == 0:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.unsqueeze(0)),
                transforms.Normalize((mean,), (std,)),
                transforms.Resize((32,32)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        elif mean.ndim == 1 and std.ndim == 1:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.t().unsqueeze(-1)),
                transforms.Normalize(mean, std),
                transforms.Lambda(lambda x: x.permute(2, 1, 0)),
                transforms.Resize((32,32)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        else:
            raise ValueError(f"mean and std are unacceptable.")
        
    def __call__(self, x):
        if self.num_aug == 0:
            return [self.transform(x)]
        
        H, W = x.shape
        device = x.device
        y = []
        for aug_i in range(self.num_aug+1):
            if self.aug_row >= H:
                row_mask = torch.ones(H, dtype=torch.bool, device=device)
            else:
                from_row_0 = False
                row_mask = torch.zeros(H, dtype=torch.bool, device=device)

                if from_row_0:
                    row_mask[0:self.aug_row] = True
                else: 
                    row_start = torch.randint(
                        low=0, 
                        high=H-self.aug_row+1, 
                        size=(1,), 
                        device=device
                        ).item()
                    row_mask[row_start:row_start+self.aug_row]=True

            if self.aug_col >= W:
                col_mask = torch.ones(W, dtype=torch.bool, device=device)
            else:
                contiguous = False
                col_mask = torch.zeros(W, dtype=torch.bool, device=device)
                if contiguous:
                    col_start = torch.randint(
                        low=0, 
                        high=W - self.aug_col + 1, 
                        size=(1,), 
                        device=device
                        ).item()
                    col_mask[col_start:col_start + self.aug_col] = True
                else:
                    sel_cols = torch.randperm(W, device=device)[:self.aug_col]
                    col_mask[sel_cols] = True

            mask2d = row_mask[:, None] & col_mask[None, :]
            x_aug = x * mask2d
            y.append(self.transform(x_aug))

        return y


def resnet20(args, feat_dim=32, mean=None, std=None, **kwargs):
    n = 3
    model = ResNet(args, BasicBlock, [n, n, n], feat_dim=feat_dim, mean=mean, std=std, **kwargs)
    return model
