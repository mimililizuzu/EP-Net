import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from embeddings.resnet20 import resnet20

class DynamicModel(nn.Module):
    """EP-Net model: feature extractor f_theta + non-parametric prototype classifier phi_W."""
    def __init__(self, args, backbone, dataset):
        super().__init__()
        self.args = args

        self.encoder = resnet20(args, feat_dim=args.embd_dim, mean=dataset.mean, std=dataset.std)
        self.feat_dim = self.encoder.feat_dim

        cls_num = args.base_cls + args.bn_cluster - 1
        self.classifier = nn.Linear(self.feat_dim, cls_num, bias=False)  # class mean prototypes
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.feat_dim, cls_num, bias=False) for _ in range(args.extra_proto)]  # extra-prototypes
        )

        self.transform = self.encoder.transform
        self.aug_transform = self.encoder.aug_transform

        self.use_extra_proto = False  # enabled after classifier weights are replaced with prototypes

    def forward(self, x):
        z = F.normalize(self.encoder(x), p=2, dim=-1)                           # [B, d]
        logit = F.linear(z, F.normalize(self.classifier.weight, p=2, dim=-1))   # [B, C]
        if self.use_extra_proto:
            # candidate score of a class = cosine similarity with each of its prototypes (Eq. 10)
            logits = [logit] + [F.linear(z, F.normalize(fc.weight, p=2, dim=-1)) for fc in self.classifiers]
            logits = torch.stack(logits, dim=0)                                 # [1+Ep, B, C]
            best = logits.max(dim=-1).values.argmax(dim=0)                      # [B] prototype with highest score
            logit = logits[best, torch.arange(z.size(0), device=z.device)]      # [B, C]
        return self.args.fw_tau * logit

    @torch.no_grad()
    def update_fc(self, dataloader):
        """Update the classifier with class mean prototypes (Eq. 8) and extra-prototypes (Eq. 9)."""
        self.eval()
        feats, labels = [], []
        for data, lb in dataloader:
            feats.append(self.encoder(data.cuda()))
            labels.append(lb.cuda())
        feats = torch.cat(feats, dim=0)     # [N, d]
        labels = torch.cat(labels, dim=0)   # [N]

        for lb in labels.unique():
            cls_feat = feats[labels == lb]                                  # [N_c, d]
            self.classifier.weight.data[lb] = cls_feat.mean(dim=0)          # class mean prototype
            for i, center in enumerate(self.kmeans_protos(cls_feat, len(self.classifiers))):
                self.classifiers[i].weight.data[lb] = center                # i-th extra-prototype

    def kmeans_protos(self, embeddings, n):
        """Inner-class clustering: Ep K-Means centers of one class as extra-prototypes."""
        if embeddings.size(0) < n:  # fewer samples than prototypes: reuse all samples
            embeddings = embeddings.repeat(n // embeddings.size(0) + 1, 1)[:n]
        km = KMeans(n_clusters=n, random_state=0).fit(embeddings.cpu().numpy())
        return torch.tensor(km.cluster_centers_, device=embeddings.device)  # [Ep, d]

    def fc_increment(self, old_model, cls_num):
        """Expand the classifier output dimension for new classes, keeping old prototypes."""
        old_weight = old_model.module.classifier.weight.data.clone()        # [C_old, d]
        self.classifier = nn.Linear(self.feat_dim, cls_num, bias=False)
        self.classifier.weight.data = torch.cat(
            [old_weight, torch.zeros(cls_num - old_weight.size(0), self.feat_dim).cuda()], dim=0)

        old_weights = [fc.weight.data.clone() for fc in old_model.module.classifiers]
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.feat_dim, cls_num, bias=False) for _ in range(len(old_weights))])
        for fc, old_weight in zip(self.classifiers, old_weights):
            fc.weight.data = torch.cat(
                [old_weight, torch.zeros(cls_num - old_weight.size(0), self.feat_dim).cuda()], dim=0)
