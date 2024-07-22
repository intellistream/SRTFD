from agents.base import ContinualLearner
from utils.setup_elements import n_classes


class MPOS_RFVL(ContinualLearner):
    def __init__(self, model, opt, params):
        super(MPOS_RFVL, self).__init__(model, opt, params)

    def train_learner(self, x_train, y_train, pseudo_x=[], pseudo_y=[], init_train=False):
        self.before_train(x_train, y_train)

        print('size: {}, {}'.format(x_train.shape, y_train.shape))
        # set up loader
        # train_dataset = dataset_transform(
        #     x_train, y_train, transform=transforms_match[self.data])
        # train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
        #                                drop_last=True)

        if init_train:
            self.model.fit(x_train, y_train)
        else:
            self.model.partial_fit(x_train, y_train)

        # # setup tracker
        # losses = AverageMeter()
        # acc_batch = AverageMeter()

        # for ep in range(self.epoch):
        #     for i, batch_data in enumerate(train_loader):
        #         # batch update
        #         batch_x, batch_y = batch_data
        #         batch_x = maybe_cuda(batch_x, self.cuda)
        #         batch_y = maybe_cuda(batch_y, self.cuda)
        #         # print(batch_x.shape)
        #         for j in range(self.mem_iters):
        #             mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

        #             if mem_x.size(0) > 0:
        #                 mem_x = maybe_cuda(mem_x, self.cuda)
        #                 mem_y = maybe_cuda(mem_y, self.cuda)
        #                 combined_batch = torch.cat((mem_x, batch_x))
        #                 combined_labels = torch.cat((mem_y, batch_y))
        #                 combined_batch_aug = self.transform(combined_batch)
        #                 features = torch.cat([self.model.forward(combined_batch).unsqueeze(
        #                     1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
        #                 loss = self.criterion(features, combined_labels)
        #                 losses.update(loss, batch_y.size(0))
        #                 self.opt.zero_grad()
        #                 loss.backward()
        #                 self.opt.step()

        #         # update mem
        #         # if not init_train:
        #         self.buffer.update(batch_x, batch_y)

        #         if i % 100 == 1 and self.verbose:
        #             print(
        #                 '==>>> it: {}, avg. loss: {:.6f}, '
        #                 .format(i, losses.avg(), acc_batch.avg())
        #             )
        self.after_train()
