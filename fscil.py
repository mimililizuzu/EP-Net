import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import time
import copy
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from load import DataScheduler
from utils import *
from bncluster import BNcluster


class FSCIL_IDS():
    def __init__(self, args, dataset, model):
        self.args = args
        self.dataset = dataset

        self.model = nn.DataParallel(model, list(range(args.gpu_num))).cuda()
        self.previous_model = None

        keys = ['train_loss', 'train_acc', 'acc', 'avg_cls_acc', 'recall', 'precision', 'F1',
                'conf_mat', 'base_acc', 'new_acc', 'base_avg_cls_acc', 'new_avg_cls_acc', 'speed']
        self.trlog = {k: [] for k in keys}

    def optim(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.lr_init,
            weight_decay=self.args.weight_dec,
            momentum=self.args.momentum,
            nesterov=True
            )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(self.args.base_epochs * ms) for ms in self.args.lr_milestone],
            gamma=self.args.lr_dec
            )
        return optimizer, scheduler

    def base_train_epoch(self, trainloader, optimizer, epoch, bn_cls):
        """Base task training with L = Lce + lambda_nce * Lnce + lambda_cp * Lcp (Eq. 7)."""
        args = self.args
        model = self.model.train()
        nce_criterion = InfoNCELoss(temperature=1 / args.ssc_tau)

        tol, total_acc, total_loss = 0, 0, 0
        tqdm_gen = tqdm(trainloader)
        for data, label in tqdm_gen:
            B = data[0].shape[0]
            data = torch.cat(data, dim=0).cuda()       # [(1+num_aug)*B, 3, 32, 32]
            label_rep = label.repeat(1 + args.num_aug).cuda()  # [(1+num_aug)*B]

            feats = F.normalize(model.module.encoder(data), dim=-1)        # [(1+num_aug)*B, d]

            # Lce: cosine similarity cross-entropy (Eq. 3)
            logits = F.linear(feats, F.normalize(model.module.classifier.weight, dim=-1)) * args.ce_tau
            ce_loss = F.cross_entropy(logits, label_rep)

            # Lnce: contrastive loss over augmented views (Eq. 5)
            nce_loss = nce_criterion(feats.reshape(1 + args.num_aug, B, -1).permute(1, 0, 2))

            # Lcp: inter-class space compression (Eq. 6), benign and attack classes separately
            cos_sim = F.linear(feats, feats)                               # [(1+num_aug)*B, (1+num_aug)*B]
            same = torch.eq(label_rep.view(-1, 1), label_rep.view(1, -1))
            pair = torch.triu(torch.ones_like(same), diagonal=1) & ~same   # inter-class pairs (i < j)
            bn = torch.isin(label_rep, torch.tensor(bn_cls).cuda())
            pair_bn = pair & bn.view(-1, 1) & bn.view(1, -1)               # benign pairs with different pseudo labels
            pair_mal = pair & ~bn.view(-1, 1) & ~bn.view(1, -1)            # attack pairs of different classes
            cp_mal = (cos_sim * pair_mal).sum() / (pair_mal.sum() + 1e-8) + 1e-10
            cp_bn = (cos_sim * pair_bn).sum() / (pair_bn.sum() + 1e-8) + 1e-10
            cp_loss = -torch.log(cp_mal) - torch.log(cp_bn) / args.bn_cluster

            loss = ce_loss + args.ssc_lamb * nce_loss + args.inter_lamb * cp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=1) == label_rep).float().mean().item()
            tol += B
            total_acc += acc * B
            total_loss += loss.item() * B
            tqdm_gen.set_description(f"session 0, epoch {epoch}, acc={total_acc/tol:.4f}, loss={total_loss/tol:.4f}")

        return total_loss / tol, total_acc / tol

    @torch.no_grad()
    def test(self, testloader, session):
        model = self.model.eval()
        all_labels, all_preds = [], []
        for data, label in tqdm(testloader):
            logits = model(data.cuda())
            pred = logits.argmax(dim=-1)
            pred = torch.clamp(pred - self.args.bn_cluster + 1, min=0)  # map pseudo labels back (Eq. 11)
            all_labels.append(label)
            all_preds.append(pred.cpu())
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        metrics = [
            accuracy_score(all_labels, all_preds),
            precision_score(all_labels, all_preds, average="macro"),
            recall_score(all_labels, all_preds, average="macro"),
            f1_score(all_labels, all_preds, average="macro")
        ]
        cm = confusion_matrix(all_labels, all_preds)

        base_acc, new_acc = None, None
        if session > 0:
            base_mask = all_labels < self.args.base_cls
            base_acc = accuracy_score(all_labels[base_mask], all_preds[base_mask])
            new_acc = accuracy_score(all_labels[~base_mask], all_preds[~base_mask])

        return metrics, base_acc, new_acc, cm, len(all_labels)

    def record(self, session, metrics, base_acc, new_acc, cm, speed, slot=None):
        """Record metrics of one session. slot=None appends, otherwise overwrites trlog[slot]."""
        cls_acc_list = cls_acc(cm)
        avg_cls = float('%.4f' % (sum(cls_acc_list) / len(cls_acc_list)))
        if session == 0:
            base_avg, new_avg = avg_cls, 0.0
        else:
            base_avg = float('%.4f' % (sum(cls_acc_list[:self.args.base_cls]) / self.args.base_cls))
            new_avg = float('%.4f' % (sum(cls_acc_list[self.args.base_cls:]) / (len(cls_acc_list) - self.args.base_cls)))

        vals = {'acc': metrics[0], 'precision': metrics[1], 'recall': metrics[2], 'F1': metrics[3],
                'conf_mat': cm, 'speed': speed, 'base_acc': base_acc, 'new_acc': new_acc,
                'avg_cls_acc': avg_cls, 'base_avg_cls_acc': base_avg, 'new_avg_cls_acc': new_avg}
        for k, v in vals.items():
            if slot is None:
                self.trlog[k].append(v)
            else:
                self.trlog[k][slot] = v

    def fmt_result(self, session, metrics, base_acc, new_acc, cm, num, ts_time):
        lines = [
            f"session {session} TEST. Acc={metrics[0]:.4f}, Precision={metrics[1]:.4f}, Recall={metrics[2]:.4f}, F1={metrics[3]:.4f}",
        ]
        if session > 0:
            lines.append(f"Base. Acc={base_acc:.4f}, New. Acc={new_acc:.4f}")
        lines.append(f"testing finished in {ts_time:.4f}s, detecting {num/ts_time:.4f} flows/s")
        lines.append("Confusion Matrix:")
        lines.append(np.array2string(cm_num_to_prob(cm), separator=', '))
        cls_acc_list = cls_acc(cm)
        for c, acc in enumerate(cls_acc_list):
            lines.append(f"class {c} acc = {acc:.4f}")
        lines.append(f"class wise average acc = {sum(cls_acc_list)/len(cls_acc_list):.4f}")
        return lines

    def run(self):
        args = self.args
        outlist = []

        def log(lines):
            if isinstance(lines, str):
                lines = [lines]
            for line in lines:
                print(line)
                outlist.append(str(line))

        tol_start_t = time.time()
        log('-------------START TRAINING-------------')

        data_scheduler = DataScheduler(
            args, self.dataset,
            tr_trans=self.model.module.aug_transform,
            ts_trans=self.model.module.transform
            )

        for session in range(args.session):
            ses_time = time.time()
            if session == 0:
                trainloader, testloader = data_scheduler.get_dataloader(session, list(range(args.base_cls)))
                log(f"session 0, base classes: {list(range(args.base_cls))}")

                # benign class division: unsupervised fine-grained pseudo labels (Eq. 2)
                trainloader.dataset.target = BNcluster(args, trainloader.dataset.data, trainloader.dataset.target)
                bn_cls = list(range(args.bn_cluster))

                optimizer, scheduler = self.optim()
                tr_start_t = time.time()
                for epoch in range(args.base_epochs):
                    epo_time = time.time()
                    loss, acc = self.base_train_epoch(trainloader, optimizer, epoch, bn_cls)
                    scheduler.step()
                    self.trlog['train_loss'].append(loss)
                    self.trlog['train_acc'].append(acc)
                    log(f"epoch:{epoch:03d}, loss:{loss:.5f}, train_acc:{acc:.5f}, epoch time:{time.time()-epo_time:.2f}s")
                log(f"base training finished in {time.time()-tr_start_t:.4f}s")

                log('-------------TESTING-------------')
                ts_start_t = time.time()
                metrics, _, _, cm, num = self.test(testloader, session)
                ts_time = time.time() - ts_start_t
                self.record(session, metrics, metrics[0], 0.0, cm, num / ts_time)
                log(self.fmt_result(session, metrics, None, None, cm, num, ts_time))

                # replace classifier weights with class mean prototypes and extra-prototypes (Eq. 8, 9)
                log('Replace the fc with protos...')
                replace_loader = torch.utils.data.DataLoader(
                    dataset=trainloader.dataset,
                    batch_size=args.base_train_batch_size,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True
                    )
                replace_loader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(replace_loader)
                self.model.module.use_extra_proto = True

                log("TEST after replacing classifier weights with protos")
                ts_start_t = time.time()
                metrics, _, _, cm, num = self.test(testloader, session)
                ts_time = time.time() - ts_start_t
                cls_acc_list = cls_acc(cm)
                if sum(cls_acc_list) / len(cls_acc_list) > self.trlog['avg_cls_acc'][0]:
                    self.record(session, metrics, metrics[0], 0.0, cm, num / ts_time, slot=0)
                log(self.fmt_result(session, metrics, None, None, cm, num, ts_time))
            else:
                cls_list = data_scheduler.cls_remain[:args.N]
                trainloader, testloader = data_scheduler.get_dataloader(session, cls_list)
                log(f"session {session}, new classes: {cls_list}")

                log('-------------UPDATING-------------')
                up_start_t = time.time()
                self.model.module.fc_increment(self.previous_model, len(data_scheduler.cls_encountered) + args.bn_cluster - 1)
                trainloader.dataset.target = trainloader.dataset.target + args.bn_cluster - 1
                self.model.module.update_fc(trainloader)
                log(f"updating finished in {time.time()-up_start_t:.4f}s")

                log('-------------TESTING-------------')
                ts_start_t = time.time()
                metrics, base_acc, new_acc, cm, num = self.test(testloader, session)
                ts_time = time.time() - ts_start_t
                self.record(session, metrics, base_acc, new_acc, cm, num / ts_time)
                log(self.fmt_result(session, metrics, base_acc, new_acc, cm, num, ts_time))

            self.previous_model = copy.deepcopy(self.model)
            log(f"session {session} finished in {time.time()-ses_time:.2f}s")

        t = self.trlog
        log('-------------ALL SESSIONS FINISHED-------------')
        log(f"Total\nAcc: {t['acc']}")
        log(f"Class-wise Avg Acc: {t['avg_cls_acc']}")
        log(f"Precision: {t['precision']}")
        log(f"Recall: {t['recall']}")
        log(f"F1: {t['F1']}")
        log(f"Base\nAcc: {t['base_acc']}")
        log(f"Class-wise Avg Acc: {t['base_avg_cls_acc']}")
        log(f"New\nAcc: {t['new_acc']}")
        log(f"Class-wise Avg Acc: {t['new_avg_cls_acc']}")
        log(f"Avg metrics of all sessions: \nacc: {sum(t['acc'])/len(t['acc'])}, prec: {sum(t['precision'])/len(t['precision'])}, recall: {sum(t['recall'])/len(t['recall'])}, F1: {sum(t['F1'])/len(t['F1'])}")
        log(f"PD: {t['acc'][0]-t['acc'][-1]}")
        log(f"Detection speed: {t['speed']} flows/s")
        log(f"Total time used: {time.time()-tol_start_t}s")

        return outlist, self.trlog


class InfoNCELoss(nn.Module):
    """Contrastive loss over augmented views of the same sample (Eq. 5)."""
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        B, V, _ = features.shape                           # [B, 1+num_aug, d]
        feats = features.reshape(B * V, -1)                # [B*(1+num_aug), d]

        sim = torch.matmul(feats, feats.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(B * V, dtype=torch.bool, device=feats.device)
        exp_sim = torch.exp(sim).masked_fill(self_mask, 0)

        # positive pairs: views augmented from the same original sample
        pos_mask = torch.eye(B, dtype=torch.bool, device=feats.device).repeat(V, V) & ~self_mask

        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True))
        loss = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1)
        return loss.mean()
