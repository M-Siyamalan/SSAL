import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, recall_score
from sklearn.metrics import precision_score, matthews_corrcoef, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from MyResNet import MyResNet
from MyMobileNet import *
from MyDenseNet import *
from DataLoader import get_datasets
from Util import *
import numpy as np
from torch.autograd import Variable
from fast_pytorch_kmeans import KMeans
from pytorch_lightning import seed_everything
import argparse
from torch_ema import ExponentialMovingAverage
import torchutils as tu
import pandas as pd
import os
import time
from datetime import timedelta

class CNNTrain(nn.Module):
    def __init__(self, args):
        super(CNNTrain, self).__init__()
        self.args = argparse.Namespace(**vars(args))
        seed_everything(args.seed)
        self.device = torch.device(args.GPU_id)
        self.trainset, self.unlabeledset, self.testloader = get_datasets(self.args.dataset, args.seed,
                                                                        pTe = self.args.pTe,
                                                                        pLa=self.args.pL,
                                                                        bs=self.args.bs)
        self.update_loaders()
        self.nclasses = len(set(self.trainset.lbls))
        self.initialize_network()


    def initialize_network(self):
        print("Initializing network, critetian, optimizer and scheduler")
        if self.args.ModelName == 'DenseNet':
            self.net = MyDenseNet(self.nclasses).to(self.device)
        elif self.args.ModelName == 'resnet18':
            self.net = MyResNet(self.args.ModelName, self.nclasses).to(self.device)
        elif self.args.ModelName == 'mobilenet':
            self.net = MyMobileNet(self.args.ModelName, self.nclasses).to(self.device)
        elif self.args.ModelName == 'ShuffleNet':
            self.net = ShuffleNet(self.args.ModelName, self.nclasses).to(self.device)
        cw = calWeights(self.trainloader.dataset.lbls)
        print("Class Weights:", cw)
        cw = torch.tensor(cw, dtype=torch.float32).cuda(self.args.GPU_id)
        self.criterion = nn.CrossEntropyLoss(weight=cw).cuda(self.args.GPU_id)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=5e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.n_epochs, eta_min=self.args.lr / 100)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=0.999)

        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"Learnable Parameters: {total_params / 1e6:.3f} M")

    def update_loaders(self):
        print('Updating Data Loaders')
        self.trainloader = torch.utils.data.DataLoader(self.trainset,
                                                       batch_size=self.args.bs,
                                                       shuffle=True,
                                                       num_workers=8)

        self.unlabeledloader = None
        if self.unlabeledset:
            self.unlabeledloader = torch.utils.data.DataLoader(self.unlabeledset,
                                                               batch_size=self.args.bs,
                                                               shuffle=False,
                                                               num_workers=8)
        printStatDataloaders(self.trainloader, self.unlabeledloader, self.testloader)


    def update_datasets_with_selected(self, selected_indices, fnIn):
        fn_all = np.array(self.unlabeledset.fnArr)
        gt_all = np.array(self.unlabeledset.lbls)
        fn_selected = fn_all[selected_indices.cpu().numpy()]
        y_t_selected = gt_all[selected_indices.cpu().numpy()]

        # only for testing
        n = 0
        for i in range(len(fn_all)):
            if fn_all[i]!=fnIn[i]:
                print(fn_all[i], fnIn[i])
                n = n+1
        print(n)
        if n!=0:
            exit()

        # print('newly selected data = ', len(y_t_selected))
        # for l in self.uniqueLbls:
        #     print('%1d : %6d' % (l, (y_t_selected == l).sum()))
        self.trainset.fnArr.extend(fn_selected)
        self.trainset.lbls.extend(y_t_selected)

        ind_not = torch.ones(len(fn_all))
        ind_not[selected_indices] = 0
        ind_not = torch.nonzero(ind_not, as_tuple=True)[0]
        self.unlabeledset.fnArr = fn_all[ind_not.cpu().numpy()]
        self.unlabeledset.lbls = gt_all[ind_not.cpu().numpy()]

        self.update_loaders()

    def train_epoch(self):
        self.net.train()
        train_loss = 0
        all_logits, all_targets = [], []
        for _, inputs, targets, _ in self.trainloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.net(inputs)  # Unpack the tuple
            loss = self.criterion(logits, targets)  # Use logits for loss calculation
            loss.backward()
            self.optimizer.step()
            if self.args.useEMA:
                self.ema.update()
            train_loss += loss.item()
            all_logits.append(logits)
            all_targets.append(targets)
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        scores = self.get_scores(all_logits, all_targets)
        return train_loss / len(self.trainloader), scores

    def test(self, loader):
        self.net.eval()
        all_logits, all_targets, all_filepaths, all_fea = [], [], [], []
        test_loss = 0
        with torch.no_grad():
            for inputs, _, targets, filepaths in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.args.useEMA:
                    with self.ema.average_parameters():
                        logits, fea = self.net(inputs)
                else:
                    logits, fea = self.net(inputs)
                loss = self.criterion(logits, targets)  # Use logits for loss calculation
                test_loss += loss.item()

                all_logits.append(logits)
                all_targets.append(targets)
                all_filepaths.extend(filepaths)
                all_fea.append(fea)

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        all_fea = torch.cat(all_fea)
        all_probs = torch.softmax(all_logits, dim=1)
        test_loss /= len(loader)
        re = {'all_probs':all_probs,
              'all_targets':all_targets,
              'all_fea':all_fea,
              'all_filepaths':all_filepaths,
              }
        scores = self.get_scores(all_logits, all_targets)
        _, pl = torch.max(all_logits, dim=1)
        return test_loss, scores, re

    def get_scores(self, outputs, targets):
        classes = np.arange(self.nclasses)
        acc, f1, pr_auc, roc_auc, mcc = 0,0,0,0,0
        nclassPredict = len(torch.unique(targets))
        outputs = outputs.clone().detach()
        _, predicted = outputs.max(1)
        predicted = predicted.cpu().numpy()
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()
        recall = recall_score(targets, predicted, average='macro') * 100
        pre = precision_score(targets, predicted, average='macro') * 100
        f1 = f1_score(targets, predicted, average='macro', labels=classes) * 100
        # bacc = balanced_accuracy_score(targets, predicted) * 100
        acc = accuracy_score(targets, predicted) * 100
        mcc = matthews_corrcoef(targets, predicted)

        if nclassPredict == self.nclasses:
            targets_bin = label_binarize(targets, classes=classes)
            pr_auc = average_precision_score(targets_bin, outputs, average="macro")
            roc_auc = roc_auc_score(targets_bin, outputs, average="macro", multi_class="ovr", labels=classes)

        scores = {'acc': acc,
                  'precision': pre,
                  'recall': recall,
                  'f1': f1,
                  'pr_auc': pr_auc,
                  'roc_auc':roc_auc,
                  'mcc':mcc
              }
        return scores

    def printStats(self, y_all, y_s, pl_s):
        y_all = y_all.to(self.device)
        pl_s = pl_s.to(self.device)
        unique_lbls = torch.unique(y_all)
        print('lbl \t tot \t selected \t acc')
        for lbl in unique_lbls:
            idx = pl_s == lbl
            tot = torch.sum(y_all==lbl).data.item()
            selected = torch.sum(idx).data.item()
            acc = accuracy_score(y_s[idx].cpu(), pl_s[idx].cpu()) * 100
            print('%1d\t%5d\t%5d\t%2.2f%%' % (lbl, tot, selected, acc))
        print()


    def SSL_update(self):
        print('\nSSL')
        print(10 * '-')
        _, _, re = self.test(self.unlabeledloader)
        probs = re['all_probs']
        lbls = re['all_targets']
        filepaths = re['all_filepaths']
        max_probs, pseudo_labels = probs.max(1)
        high_conf_mask = max_probs > self.args.thr

        new_fnArr = [fp for fp, mask in zip(filepaths, high_conf_mask) if mask]
        new_labels = pseudo_labels[high_conf_mask].tolist()
        self.printStats(lbls, lbls[high_conf_mask], pseudo_labels[high_conf_mask])
        self.trainset.fnArr.extend(new_fnArr)
        self.trainset.lbls.extend(new_labels)

        keep_mask = ~high_conf_mask
        self.unlabeledset.fnArr = [fp for fp, mask in zip(filepaths, keep_mask) if mask]
        self.unlabeledset.lbls = lbls[keep_mask].tolist()
        self.update_loaders()

        print("Dataset statistics after SSL:")
        self.print_dataset_stats()

    def AL_update(self):
        print('\nAL')
        print(20 * '-')
        _, _, re = self.test(self.unlabeledloader)
        probs = re['all_probs']
        fea = re['all_fea']
        filepaths = re['all_filepaths']

        if len(self.unlabeledset.fnArr) > self.args.b:
            selected_indices = self.clusterAndSelect(self.args.b, fea, probs)
        else:
            selected_indices = list(range(len(self.unlabeledset.fnArr)))
            selected_indices = torch.IntTensor(selected_indices).cuda(self.args.GPU_id)
        self.update_datasets_with_selected(selected_indices, filepaths)

        print("Dataset statistics after AL:")
        self.print_dataset_stats()

    def clusterAndSelect(self, b, x, prob):
        print('Feature size for KMeans: ', x.shape, x.device)
        print('Kmeans ...', end='')
        kmeans = KMeans(n_clusters=b, mode='euclidean', init_method="kmeans++", verbose=1)
        cluster_ids = kmeans.fit_predict(x)
        print('Done')

        idx_selected = []
        for k in range(0, b):
            ind_k = torch.nonzero(cluster_ids == k, as_tuple=True)[0]
            if ind_k.nelement() > 0:
                #ind_tmp2 = self.selectData_LConfidence(prob[ind_k, :], 1)
                ind_tmp2 = self.selectData_LMargin(prob[ind_k, :], 1)
                idx_selected.append(ind_k[ind_tmp2[0]])
        idx_selected = torch.stack(idx_selected, 0)

        if len(idx_selected) < b:
            print('selecting additional', b - len(idx_selected), 'from', len(prob))
            prob[idx_selected] = float('inf')  # Ensures selected indices aren't reselected
            _, addind = torch.topk(prob, min(len(prob), b - len(idx_selected)), largest=False)
            if addind.dim() == 1:  # Ensure dimensions match for concatenation
                addind = addind.unsqueeze(1)
            idx_selected = torch.cat([idx_selected, addind], dim=0)

        return idx_selected

    def selectData_LMargin(self, prob, b):
        topk, _ = torch.topk(prob, k=2, dim=1)
        _, ind = torch.sort(topk[:, 0] - topk[:, 1], descending=False)
        ind = ind[:b]
        return ind

    def selectData_LConfidence(self, prob, b):
        prob, yp = torch.max(prob, dim=1)
        _, ind = torch.sort(prob, descending=False)
        ind = ind[:b]
        return ind

    def selectDataKCenterGREEDY(self, xu, b):
        _, _, logitDetail = self.test(self.loader_L, True)
        xl = logitDetail['fea']
        ind_u = torch.arange(0, xu.shape[0]).long()
        ind_s = []
        start_time = time.time()
        for i in range(b):
            d = torch.cdist(xu[ind_u, :], xl, p=2)
            _, si = torch.max(torch.min(d, dim=1)[0], 0)
            si_ori = ind_u[si]
            ind_s.append(si_ori)
            xl = torch.cat((xl, xu[si_ori, :].view(1, -1)), dim=0)
            ind_u = torch.cat((ind_u[:si], ind_u[si + 1:]), dim=0)
            if i % 100 == 0: print(i, end=',')
        print(' \n ', time.time() - start_time)
        ind_s = torch.stack(ind_s, 0)
        return ind_s

    def print_dataset_stats(self):
        def getCount(y):
            count = [0] * self.nclasses
            for lbl in y:
                count[lbl] += 1
            return count

        count_Tr = getCount(self.trainset.lbls)
        count_Te = getCount(self.testloader.dataset.lbls)
        count_UL = getCount(self.unlabeledset.lbls)

        print("lbl \t Label \t Unlbl \t Test")
        for i in range(self.nclasses):
            print(f"{i}\t {count_Tr[i]} \t    {count_UL[i]} \t {count_Te[i]}")
        print(f" \t {sum(count_Tr)} \t  {sum(count_UL)} \t {sum(count_Te)}")
        print()

    def cnnTrainAndTest(self):
        printStatDataloaders(self.trainloader, self.unlabeledloader, self.testloader)
        self.initialize_network()
        results = []
        print('Epoch    TrLoss   TeLoss    TeAcc     TeMAC     TeF1')
        for epoch in range(self.args.n_epochs):
            lr = tu.get_lr(self.optimizer)
            train_loss, scores_Tr = self.train_epoch()
            test_loss, scores,_ = self.test(self.testloader)
            self.scheduler.step()
            results.append([scores['acc'], scores['precision'], scores['recall'], scores['f1'], scores['pr_auc'], scores['roc_auc'], scores['mcc']])
            print('%2d \t %1.5f |\t %7.2f \t %2.2f%% \t|| %7.2f \t %2.2f%% \t %2.2f%% \t %2.2f%% \t %2.2f%% \t %2.2f \t %2.2f \t %2.2f' % (
            epoch, lr, train_loss, scores_Tr['acc'], test_loss, scores['acc'], scores['precision'], scores['recall'], scores['f1'], scores['pr_auc'],
            scores['roc_auc'], scores['mcc']))

        return np.mean(results[-5:], axis=0)  # Return average of last 5 epochs

    def saveResults(self, itr):
        _, _, re = self.test(self.testloader)
        print(re["all_probs"].shape)
        df = pd.DataFrame({
            "filename": re["all_filepaths"],
            "y_true": re["all_targets"].cpu().numpy().ravel(),
            "y_pred": re["all_probs"].cpu().numpy().tolist()
        })
        # Optionally save to CSV
        fn = os.path.join(self.args.dirname, 'pL_' + str(int(self.args.pL*100)) + '_SSL' + str(int(self.args.useSSL)) +
                          '_AL' + str(int(self.args.useAL)) + '_itr' + str(itr) + '_predictions.pkl')
        print(fn)
        # df.to_csv(fn, index=False)
        df.to_pickle(fn)

    def iterate(self):
        results = []
        scores = self.cnnTrainAndTest()
        results.extend(scores)
        self.saveResults(0)

        if (self.args.useSSL or self.args.useAL) and self.unlabeledset is not None:
            for iteration in range(self.args.num_ite):
                print(f"\nIteration {iteration + 1} of {self.args.num_ite}")
                print(20*'-')
                no_unlbl = len(self.unlabeledset.fnArr)
                if no_unlbl > 0 and self.args.useSSL:
                    print(len(self.unlabeledset.fnArr), len(self.unlabeledset.lbls))
                    self.SSL_update()

                if no_unlbl > 0 and self.args.useAL:
                    self.AL_update()

                scores = self.cnnTrainAndTest()
                results.extend(scores)
                print(results)
                # self.saveResults(iteration+1)

        self.saveResults(self.args.num_ite)
        desc = ['acc', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'mcc']

        return results, desc