from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils.kd_manager import KdManager
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import copy
from utils.loss import SupConLoss
import pickle
# from models.FocalLoss import FocalLoss
from utils.con_m import conf_matrix
from utils.setup_elements import n_classes
from sklearn.metrics import recall_score
from sklearn import metrics
from utils.recall_loss import RecallLoss, FocalLoss


class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):
        super(ContinualLearner, self).__init__()
        self.params = params
        self.model = model
        self.opt = opt
        self.data = params.data
        self.cuda = params.cuda
        self.epoch = params.epoch
        self.batch = params.batch
        self.verbose = params.verbose
        self.old_labels = []
        self.new_labels = []
        self.task_seen = 0
        self.kd_manager = KdManager()
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}
        self.class_task_map = {}

        self.pseudo_x = []
        self.pseudo_y = []

    def before_train(self, x_train, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i

        for i in new_labels:
            self.class_task_map[i] = self.task_seen

    @abstractmethod
    def train_learner(self, x_train, y_train):
        pass

    def after_train(self):
        # self.old_labels = list(set(self.old_labels + self.new_labels))
        self.old_labels += self.new_labels
        self.new_labels_zombie = copy.deepcopy(self.new_labels)
        self.new_labels.clear()
        self.task_seen += 1
        if self.params.trick['review_trick'] and hasattr(self, 'buffer'):
            self.model.train()
            mem_x = self.buffer.buffer_img[:self.buffer.current_index]
            mem_y = self.buffer.buffer_label[:self.buffer.current_index]
            # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if mem_x.size(0) > 0:
                rv_dataset = TensorDataset(mem_x, mem_y)
                rv_loader = DataLoader(rv_dataset, batch_size=self.params.eps_mem_batch, shuffle=True, num_workers=0,
                                       drop_last=True)
                for ep in range(1):
                    for i, batch_data in enumerate(rv_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        logits = self.model.forward(batch_x, batch_y)
                        if self.params.agent == 'SCR':
                            logits = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                                self.model.forward(self.transform(batch_x)).unsqueeze(1)], dim=1)
                        loss = self.criterion(logits, batch_y)
                        self.opt.zero_grad()
                        loss.backward()
                        params = [p for p in self.model.parameters(
                        ) if p.requires_grad and p.grad is not None]
                        grad = [p.grad.clone()/10. for p in params]
                        for g, p in zip(grad, params):
                            p.grad.data.copy_(g)
                        self.opt.step()

        if self.params.trick['kd_trick'] or self.params.agent == 'LWF':
            self.kd_manager.update_teacher(self.model)

    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        # FC = FocalLoss(n_classes[self.params.data])
        RL = FocalLoss()
        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ['SCR', 'SCP']:
            SC = SupConLoss(temperature=self.params.temp)
            return SC(logits, labels)
        else:
            loss_ce = ce(logits, labels)
            # loss_FC = FC(logits, labels)
            loss_re = RL(logits, labels)
            return loss_re

    def forward(self, x):
        return self.model.forward(x)

    def evaluate(self, test_loaders, curr_task, curr_run, conf_threshold=0.95, uncertain_threshold=0.05):
        if self.params.agent != 'MPOS_RVFL':
            self.model.eval()
        acc_array = np.zeros(len(test_loaders))
        recall = np.zeros(len(test_loaders))
        p = np.zeros(len(test_loaders))
        f1 = np.zeros(len(test_loaders))
        g_mean = np.zeros(len(test_loaders))
        support_micro = np.zeros(len(test_loaders))

        if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
            exemplar_means = {}
            cls_exemplar = {cls: [] for cls in self.old_labels}
            buffer_filled = self.buffer.current_index
            for x, y in zip(self.buffer.buffer_img[:buffer_filled], self.buffer.buffer_label[:buffer_filled]):
                cls_exemplar[y.item()].append(x)
            for cls, exemplar in cls_exemplar.items():
                features = []
                # Extract feature for each exemplar in p_y
                for ex in exemplar:
                    feature = self.model.features(
                        ex.unsqueeze(0)).detach().clone()
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()  # Normalize
                    features.append(feature)
                if len(features) == 0:
                    mu_y = maybe_cuda(torch.normal(0, 1, size=tuple(
                        self.model.features(x.unsqueeze(0)).detach().size())), self.cuda)
                    mu_y = mu_y.squeeze()
                else:
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means[cls] = mu_y
        with torch.no_grad():
            if self.params.error_analysis:
                error = 0
                no = 0
                nn = 0
                oo = 0
                on = 0
                new_class_score = AverageMeter()
                old_class_score = AverageMeter()
                correct_lb = []
                predict_lb = []

            count = 0

            self.pseudo_x = []
            self.pseudo_y = []
            full_acc = torch.empty(0)
            full_label = torch.empty(0)
            for task, test_loader in enumerate(test_loaders):
                acc = AverageMeter()
                accuracy11 = torch.empty(0)
                Label11 = torch.empty(0)
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
                        # (batch_size, feature_size)
                        feature = self.model.features(batch_x)

                        for j in range(feature.size(0)):  # Normalize
                            feature.data[j] = feature.data[j] / \
                                feature.data[j].norm()
                        # (batch_size, feature_size, 1)
                        feature = feature.unsqueeze(2)
                        # (n_classes, feature_size)
                        means = torch.stack([exemplar_means[cls]
                                            for cls in self.old_labels])

                        # old ncm
                        # (batch_size, n_classes, feature_size)
                        means = torch.stack([means] * batch_x.size(0))
                        means = means.transpose(1, 2)
                        # (batch_size, feature_size, n_classes)
                        feature = feature.expand_as(means)
                        # (batch_size, n_classes)
                        dists = (feature - means).pow(2).sum(1).squeeze()
                        _, pred_label = dists.min(1)
                        # may be faster
                        # feature = feature.squeeze(2).T
                        # _, preds = torch.matmul(means, feature).max(0)
                        correct_cnt = (np.array(self.old_labels)[
                            pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                    else:
                        if self.params.agent == 'MPOS_RVFL':
                            pred_label = torch.from_numpy(
                                self.model.predict(batch_x.cpu().numpy()))
                        else:
                            logits = self.model.forward(batch_x)
                            probs = F.softmax(logits, dim=1)
                            conf, pred_label = torch.max(probs, 1)

                        if curr_task == task and self.params.agent == 'SRTFD':
                            self.model.train()
                            predictions = []
                            for _ in range(5):
                                with torch.no_grad():
                                    logits = self.model.forward(batch_x)
                                    probs = F.softmax(logits, dim=-1)
                                    predictions.append(probs.unsqueeze(0))
                            predictions = torch.cat(predictions, 0)
                            mean_prediction = predictions.mean(0)
                            conf, pred_label = torch.max(
                                mean_prediction, dim=-1)
                            uncertainty = - \
                                torch.sum(
                                    mean_prediction * torch.log(mean_prediction + 1e-10), dim=-1)

                            for i in range(len(batch_x)):
                                # print(conf[i], uncertainty[i])
                                if conf[i] > 0.95 and uncertainty[i] < 2:
                                    # print('Here')
                                    self.pseudo_x.append(batch_x[i].cpu())
                                    self.pseudo_y.append(batch_y[i].cpu())
                                if conf[i] < 0.45 and uncertainty[i] < 2:
                                    # print('Here')
                                    self.pseudo_x.append(batch_x[i].cpu())
                                    self.pseudo_y.append(batch_y[i].cpu())

                            # print(len(self.pseudo_x))
                        accuracy11 = maybe_cuda(accuracy11, self.cuda)
                        pred_label = maybe_cuda(pred_label, self.cuda)
                        accuracy11 = torch.cat((accuracy11, pred_label), dim=0)
                        # print(len(accuracy))
                        correct_cnt = (pred_label == batch_y).sum(
                        ).item()/batch_y.size(0)

                        Label11 = maybe_cuda(Label11, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        Label11 = torch.cat((Label11, batch_y), dim=0)
                        
                        

                    if self.params.error_analysis:
                        correct_lb += [task] * len(batch_y)
                        for i in pred_label:
                            predict_lb.append(self.class_task_map[i.item()])
                        if task < self.task_seen-1:
                            # old test
                            total = (pred_label != batch_y).sum().item()
                            wrong = pred_label[pred_label != batch_y]
                            error += total
                            on_tmp = sum([(wrong == i).sum().item()
                                         for i in self.new_labels_zombie])
                            oo += total - on_tmp
                            on += on_tmp
                            old_class_score.update(logits[:, list(set(
                                self.old_labels) - set(self.new_labels_zombie))].mean().item(), batch_y.size(0))
                        elif task == self.task_seen - 1:
                            # new test
                            total = (pred_label != batch_y).sum().item()
                            error += total
                            wrong = pred_label[pred_label != batch_y]
                            no_tmp = sum([(wrong == i).sum().item() for i in list(
                                set(self.old_labels) - set(self.new_labels_zombie))])
                            no += no_tmp
                            nn += total - no_tmp
                            new_class_score.update(
                                logits[:, self.new_labels_zombie].mean().item(), batch_y.size(0))
                        else:
                            pass
                    acc.update(correct_cnt, batch_y.size(0))
                acc_array[task] = acc.avg()
                # recall1[task] = recall_score(
                    # accuracy11.cpu(), Label11.cpu(), average='macro', zero_division=0)
                p[task], recall[task], f1[task], _ = metrics.precision_recall_fscore_support(
                    accuracy11.cpu(), Label11.cpu(), zero_division=0, average='macro')

                # p_mean = p.mean()
                # recall_mean = recall.mean()

                # f1[task] = 2 * p_mean * recall_mean / (p_mean + recall_mean)

                # g_mean[task] = np.sqrt(p_mean * recall_mean)
                g_mean[task] = np.sqrt(p[task] * recall[task])

                full_acc = maybe_cuda(full_acc, self.cuda)
                accuracy11 = maybe_cuda(accuracy11, self.cuda)
                full_label = maybe_cuda(full_label, self.cuda)
                Label11 = maybe_cuda(Label11, self.cuda)
                full_acc = torch.cat((full_acc, accuracy11), dim=0)
                full_label = torch.cat((full_label, Label11), dim=0)

            # print('recall {}, f1 {}, g_mean {}'.format(
            #     np.mean(recall), np.mean(f1), np.mean(g_mean)))
            conf_matrix(full_acc, full_label, f'{self.params.data}_{self.params.agent}_{
                        self.params.cl_type}_run_{curr_run}_task_{curr_task}.png')
            # print(recall,f1)

        print(acc_array)

       # print(len(accuracy11))
        if self.params.error_analysis:
            self.error_list.append((no, nn, oo, on))
            self.new_class_score.append(new_class_score.avg())
            self.old_class_score.append(old_class_score.avg())
            print("no ratio: {}\non ratio: {}".format(
                no/(no+nn+0.1), on/(oo+on+0.1)))
            print(self.error_list)
            print(self.new_class_score)
            print(self.old_class_score)
            self.fc_norm_new.append(
                self.model.linear.weight[self.new_labels_zombie].mean().item())
            self.fc_norm_old.append(self.model.linear.weight[list(
                set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            self.bias_norm_new.append(
                self.model.linear.bias[self.new_labels_zombie].mean().item())
            self.bias_norm_old.append(self.model.linear.bias[list(
                set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            print(self.fc_norm_old)
            print(self.fc_norm_new)
            print(self.bias_norm_old)
            print(self.bias_norm_new)
            with open('confusion', 'wb') as fp:
                pickle.dump([correct_lb, predict_lb], fp)
        return acc_array, recall, p, f1, g_mean
