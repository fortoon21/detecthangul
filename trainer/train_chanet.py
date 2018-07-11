import os
import time
import torch
import torch.optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from loss.ssd_loss import SSDLoss
from metrics.voc_eval import voc_eval
from modellibs.s3fd.box_coder import S3FDBoxCoder
from utils.average_meter import AverageMeter
import matlab.engine

class Trainer(object):

    def __init__(self, opt, train_dataloader, valid_dataloader, model):
        self.opt = opt
        self.current_lr = opt.lr
        self.start_epoch = opt.start_epochs

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.max_iter_train = len(self.train_dataloader)
        self.max_iter_valid = len(self.valid_dataloader)

        self.model = model

        self.criterion_first = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_middle = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_last = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_config = torch.nn.CrossEntropyLoss().cuda()

        self.optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

        self.best_loss = float('inf')

        if opt.resume:
            self.optimizer.load_state_dict(torch.load(opt.resume_path)['optimizer'])

    def train_model(self, max_epoch, learning_rate, layers=None):

        self.max_epoch = max_epoch

        for epoch in range(self.start_epoch, self.max_epoch):
            self.adjust_learning_rate(self.optimizer, epoch)

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
        for batch_idx, (inputs, labels) in enumerate(self.train_dataloader):
            label_first = labels[0]
            label_middle = labels[1]
            label_last = labels[2]
            label_config = labels[3]

            inputs = inputs.to(self.opt.device)
            label_first = label_first.to(self.opt.device)
            label_middle = label_middle.to(self.opt.device)
            label_last = label_last.to(self.opt.device)
            label_config = label_config.to(self.opt.device)

            output_first, output_middle, output_last, output_config = self.model(inputs)
            loss_first = self.criterion_first(output_first, label_first)
            loss_middle = self.criterion_middle(output_middle, label_middle)
            loss_last = self.criterion_last(output_last, label_last)
            loss_config = self.criterion_config(output_config, label_config)
            loss = loss_first + loss_middle + loss_last + loss_config
            loss.backward()
            if ((batch_idx + 1) % self.opt.accum_grad == 0) or ((batch_idx+1) == self.max_iter_train):
                self.optimizer.step()
                self.model.zero_grad()
                self.optimizer.zero_grad()

            train_loss += loss.item()
            if batch_idx % self.opt.print_freq == 0:
                print('Epoch[%d/%d] Iter[%d/%d] Learning Rate: %.6f Total Loss: %.4f, First Loss: %.4f, Middle Loss: %.4f, Last Loss: %.4f, Config Loss: %.4f' %
                      (epoch, self.max_epoch, batch_idx, self.max_iter_train, self.current_lr, loss.item(), loss_first.item(), loss_middle.item(), loss_last.item(), loss_config.item()))

    def valid_epoch(self, epoch):

        correct_f = 0
        correct_m = 0
        correct_l = 0
        correct_c = 0

        """ validate """
        self.model.eval()
        test_loss = 0

        for batch_idx, (inputs, labels) in enumerate(self.valid_dataloader):
            with torch.no_grad():
                label_first = labels[0]
                label_middle = labels[1]
                label_last = labels[2]
                label_config = labels[3]

                inputs = inputs.to(self.opt.device)
                label_first = label_first.to(self.opt.device)
                label_middle = label_middle.to(self.opt.device)
                label_last = label_last.to(self.opt.device)
                label_config = label_config.to(self.opt.device)

                output_first, output_middle, output_last, output_config = self.model(inputs)
                loss_first = self.criterion_first(output_first, label_first)
                loss_middle = self.criterion_middle(output_middle, label_middle)
                loss_last = self.criterion_last(output_last, label_last)
                loss_config = self.criterion_config(output_config, label_config)
                loss = loss_first + loss_middle + loss_last + loss_config

                pred_f = output_first.data.max(1, keepdim=True)[1].cpu()
                pred_m = output_middle.data.max(1, keepdim=True)[1].cpu()
                pred_l = output_last.data.max(1, keepdim=True)[1].cpu()
                pred_c = output_config.data.max(1, keepdim=True)[1].cpu()
                correct_f += pred_f.eq(label_first.cpu().view_as(pred_f)).sum()
                correct_m += pred_m.eq(label_middle.cpu().view_as(pred_m)).sum()
                correct_l += pred_l.eq(label_last.cpu().view_as(pred_l)).sum()
                correct_c += pred_c.eq(label_config.cpu().view_as(pred_c)).sum()

                test_loss += loss.item()

                if batch_idx % self.opt.print_freq_eval == 0:
                    print('Validation[%d/%d] Total Loss: %.4f, First Loss: %.4f, Middle Loss: %.4f, Last Loss: %.4f, Conf Loss: %.4f' %
                          (batch_idx, len(self.valid_dataloader), loss.item(), loss_first.item(), loss_middle.item(), loss_last.item(), loss_config.item()))

        num_test_data = len(self.valid_dataloader.dataset)
        accuracy_f = 100. * correct_f / num_test_data
        accuracy_m = 100. * correct_m / num_test_data
        accuracy_l = 100. * correct_l / num_test_data
        accuracy_c = 100. * correct_c / num_test_data

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
        print('Val Accuracy_F: {}/{} ({:.0f}%) | Val Accuracy_M: {}/{} ({:.0f}%) | Val Accuracy_L: {}/{} ({:.0f}%) | Val Accuracy_C: {}/{} ({:.0f}%)\n'.format(
            correct_f, num_test_data, accuracy_f,
            correct_m, num_test_data, accuracy_m,
            correct_l, num_test_data, accuracy_l,
            correct_c, num_test_data, accuracy_c))

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.current_lr = self.opt.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def make_dir(self, dir_path):
        if not os.path.exists(os.path.join(self.opt.expr_dir, dir_path)):
            os.mkdir(os.path.join(self.opt.expr_dir, dir_path))
