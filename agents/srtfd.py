import torch
import numpy as np
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
from utils.Cluster import Cluster, KL_div

from utils.coreset import CoresetGreedy


class SRTFD(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SRTFD, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters

    def train_learner(self, x_train, y_train, pseudo_x=[], pseudo_y=[], alpha = 0.7, coreset_ratio = 0.8, init_train=False):
        self.before_train(x_train, y_train)
        Beta = 0.5
        print('before size: {}, {}'.format(x_train.shape, y_train.shape))
        x_train, y_train = KL_div(x_train, y_train, self.buffer,6)
        #pseudo_x, pseudo_y = KL_div(x_train, y_train, self.buffer,6)
        print('size: {}, {}'.format(x_train.shape, y_train.shape))

        pseudo_x = np.array(pseudo_x)
        pseudo_y = np.array(pseudo_y)

        coreset = CoresetGreedy(x_train, pseudo_x, y_train, pseudo_y)
        idx = coreset.sample(coreset_ratio, self.buffer)

        orig_size = len(y_train)

        mask_l = idx < orig_size
        mask_pl = np.logical_and(idx >= len(
            y_train), idx < orig_size + len(pseudo_y))
        
        print(len(idx[mask_l]), len(idx))

        x_train = x_train[idx[mask_l]]
        y_train = y_train[idx[mask_l]]
        pseudo_x = pseudo_x[idx[mask_pl] - orig_size]
        pseudo_y = pseudo_y[idx[mask_pl] - orig_size]

        # set up loader
        train_dataset = dataset_transform(
            x_train, y_train, transform=transforms_match[self.data])
        ps_train_dataset = dataset_transform(
            pseudo_x, pseudo_y, transform=transforms_match[self.data])

        print(len(train_dataset), len(ps_train_dataset))

        if alpha is not None:
            ps_train_loader = data.DataLoader(ps_train_dataset, batch_size=self.batch, shuffle=len(ps_train_dataset) != 0, num_workers=0,
                                              drop_last=True)
        else:
            train_dataset = data.ConcatDataset(
                [train_dataset, ps_train_dataset])

        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=False)

        # set up model
        self.model = self.model.train()

        # setup tracker
        losses_batch = AverageMeter()
        losses_batch_ps = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_batch_ps = AverageMeter()
        acc_mem = AverageMeter()

        for ep in range(self.epoch):

            if alpha is not None:
                for i, batch_data_ps in enumerate(ps_train_loader):
                    # batch update
                    batch_x_ps, batch_y_ps = batch_data_ps
                    #print(batch_x.shape)

                    batch_x_ps = maybe_cuda(batch_x_ps, self.cuda)
                    batch_y_ps = maybe_cuda(batch_y_ps, self.cuda)
                    for j in range(self.mem_iters):
                        logits_ps = self.model.forward(batch_x_ps)
                        loss_ps = self.criterion(logits_ps, batch_y_ps)
                        if self.params.trick['kd_trick']:
                            loss_ps = 1 / (self.task_seen + 1) * loss_ps + (1 - 1 / (self.task_seen + 1)) * \
                                self.kd_manager.get_kd_loss(logits_ps, batch_x_ps)
                        if self.params.trick['kd_trick_star']:
                            loss_ps = 1/((self.task_seen + 1) ** 0.5) * loss_ps + \
                                (1 - 1/((self.task_seen + 1) ** 0.5)) * \
                                self.kd_manager.get_kd_loss(logits_ps, batch_x_ps)
                        loss_ps = alpha * loss_ps
                        _, pred_label_ps = torch.max(logits_ps, 1)
                        # print(pred_label, batch_y)
                        correct_cnt = (pred_label_ps == batch_y_ps).sum(
                        ).item() / batch_y_ps.size(0)
                        # update tracker
                        acc_batch_ps.update(correct_cnt, batch_y_ps.size(0))
                        losses_batch_ps.update(loss_ps, batch_y_ps.size(0))
                        # backward
                        self.opt.zero_grad()
                        loss_ps.requires_grad_(True)
                        loss_ps.backward()
                        self.opt.step()
                    if not init_train and ep == 0:
                        self.buffer.update(batch_x_ps, batch_y_ps)

            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters):
                    logits = self.model.forward(batch_x)
                    loss = self.criterion(logits, batch_y)
                    if self.params.trick['kd_trick']:
                        loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                            self.kd_manager.get_kd_loss(logits, batch_x)
                    if self.params.trick['kd_trick_star']:
                        loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                            (1 - 1/((self.task_seen + 1) ** 0.5)) * \
                            self.kd_manager.get_kd_loss(logits, batch_x)
                    if alpha is not None:
                        loss = loss
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum(
                    ).item() / batch_y.size(0)
                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0))
                    losses_batch.update(loss, batch_y.size(0))
                    # backward
                    self.opt.zero_grad()
                    loss.requires_grad_(True)
                    loss.backward()

                    # mem update
                    mem_x, mem_y = self.buffer.retrieve()
                    #print(mem_x.size(0))
                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_logits = self.model.forward(mem_x)
                        loss_mem = self.criterion(mem_logits, mem_y)
                        if self.params.trick['kd_trick']:
                            loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                self.kd_manager.get_kd_loss(mem_logits, mem_x)
                        if self.params.trick['kd_trick_star']:
                            loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                (1 - 1 / ((self.task_seen + 1) ** 0.5)) * \
                                self.kd_manager.get_kd_loss(mem_logits, mem_x)
                        # update tracker
                        losses_mem.update(loss_mem, mem_y.size(0))
                        _, pred_label = torch.max(mem_logits, 1)
                        correct_cnt = (pred_label == mem_y).sum(
                        ).item() / mem_y.size(0)
                        acc_mem.update(correct_cnt, mem_y.size(0))

                        loss_mem.requires_grad_(True)
                        loss_mem.backward()

                    self.opt.step()

                # update mem
                if not init_train and ep == 0:
                    self.buffer.update(batch_x, batch_y)

                if i % 20 == 0 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                        .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running pstrain acc: {:.3f}'
                        .format(i, losses_batch_ps.avg(), acc_batch_ps.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                        .format(i, losses_mem.avg(), acc_mem.avg())
                    )
        self.after_train()
