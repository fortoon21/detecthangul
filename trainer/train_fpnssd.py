import os
import time
import torch
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable
from loss.ssd_loss import SSDLoss
from metrics.voc_eval import voc_eval
from modellibs.fpnssd.box_coder import FPNSSDBoxCoder
from utils.average_meter import AverageMeter
from utils.box import change_box_order


class Trainer(object):

    def __init__(self, opt, train_dataloader, valid_dataloader, model):
        self.opt = opt
        self.current_lr = opt.lr

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.max_iter_train = len(self.train_dataloader)
        self.max_iter_valid = len(self.valid_dataloader)

        self.model = model
        self.box_coder = FPNSSDBoxCoder()

        self.criterion = SSDLoss(opt.num_classes)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # init metric
        self.metric = voc_eval
        self.best_loss = float('inf')

    def train_model(self, max_epoch, learning_rate, layers=None):

        self.max_epoch = max_epoch

        for epoch in range(1, self.max_epoch):
            if epoch in self.opt.lr_steps:
                self.adjust_learning_rate(self.opt.lr_decay_rate)

            self.train_epoch(epoch)

            self.valid_epoch(epoch)

        print('')
        print('optimization done')
        save_dir = 'experiments/%s_%s_%s' % (self.opt.dataset, self.opt.task, self.opt.model)
        file_name = '%s_%s_%s_best_loss_%f' % (self.opt.dataset, self.opt.task, self.opt.model, self.best_loss)
        os.rename(self.opt.expr_dir,  os.path.join(save_dir,file_name))

    def train_epoch(self, epoch):
        """ training """
        self.model.train()
        self.optimizer.zero_grad()

        train_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(self.train_dataloader):
            inputs = Variable(inputs)
            loc_targets = Variable(loc_targets)
            cls_targets = Variable(cls_targets)

            if self.opt.num_gpus:
                inputs = inputs.cuda()
                loc_targets = loc_targets.cuda()
                cls_targets = cls_targets.cuda()


            loc_preds, cls_preds = self.model(inputs)
            loss, loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            if ((batch_idx + 1) % self.opt.accum_grad == 0) or ((batch_idx+1) == self.max_iter_train):
                self.optimizer.step()
                self.model.zero_grad()
                self.optimizer.zero_grad()

            train_loss += loss.data[0]
            if batch_idx % self.opt.print_freq == 0:
                print('Epoch[%d/%d] Iter[%d/%d] Learning Rate: %.6f Total Loss: %.4f, Loc Loss: %.4f, Cls Loss: %.4f' %
                      (epoch, self.max_epoch, batch_idx, self.max_iter_train, self.current_lr,
                       loss.item(), loc_loss.item(), cls_loss.item()))

    def valid_epoch(self, epoch):

        batch_time = AverageMeter()

        total_losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()

        top1 = AverageMeter()
        top5 = AverageMeter()

        """ validate """
        self.model.eval()
        test_loss = 0
        gt_boxes = []
        gt_labels = []

        end = time.time()
        log = ''
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(self.valid_dataloader):
            with torch.no_grad():
                gt_boxes.append(loc_targets.squeeze(0))
                gt_labels.append(cls_targets.squeeze(0))

                inputs = Variable(inputs)
                loc_targets = Variable(loc_targets)
                cls_targets = Variable(cls_targets)

                if self.opt.num_gpus:
                    inputs = inputs.cuda()
                    loc_targets = loc_targets.cuda()
                    cls_targets = cls_targets.cuda()

                loc_preds, cls_preds = self.model(inputs)
                loss, loc_loss, cls_loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)

                total_losses.update(loss.data[0], inputs.data.size(0))
                loc_losses.update(loc_loss.data[0], inputs.data.size(0))
                cls_losses.update(cls_loss.data[0], inputs.data.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                test_loss += loss.data[0]

                if batch_idx % self.opt.print_freq_eval == 0:
                    print('Validation[%d/%d] Total Loss: %.4f, Loc Loss: %.4f, Cls Loss: %.4f' %
                          (batch_idx, len(self.valid_dataloader), loss.data[0], loc_loss.data[0], cls_loss.data[0]))

        test_loss /= len(self.valid_dataloader)
        if test_loss < self.best_loss:
            print('Saving..')

            state = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': test_loss,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(self.opt.expr_dir, 'model_best.pth'))
            self.best_loss = test_loss
        print('[*] Model %s,\tCurrent Loss: %f\tBest Loss: %f' % (self.opt.model, test_loss, self.best_loss))

    def adjust_learning_rate(self, gamma):

        self.current_lr = self.current_lr * gamma
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
