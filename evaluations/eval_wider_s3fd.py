import os
import torch
import time
import torch.nn.functional as F
from modellibs.s3fd.box_coder import S3FDBoxCoder
from utils.average_meter import AverageMeter
from loss.ssd_loss import SSDLoss


class Evaluator(object):

    def __init__(self, opt, test_dataloader, model):

        self.opt = opt
        self.test_dataloader = test_dataloader
        self.model = model
        self.box_coder = S3FDBoxCoder()

    def load_state_dict(self, folder_name):

        file_name = os.path.join(self.opt.proj_dir, 'experiments/wider_detection_s3fd/', folder_name, 'model_best.pth')
        self.model.load_state_dict(torch.load(file_name))

    def validate(self):

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
        for batch_idx, (inputs, loc_targets, cls_targets, fname) in enumerate(self.test_dataloader):

            img_size = [inputs.size()[2] * 1.0, inputs.size()[3] * 1.0]
            # img_size = img_size.numpy()
            box_coder = S3FDBoxCoder(input_size=img_size)

            pred_boxes = []
            pred_labels = []
            pred_scores = []

            with torch.no_grad():
                gt_boxes.append(loc_targets.squeeze(0))
                gt_labels.append(cls_targets.squeeze(0))

                inputs = inputs.to(self.opt.device)
                loc_targets = loc_targets.to(self.opt.device)
                cls_targets = cls_targets.to(self.opt.device)

                loc_preds, cls_preds = self.model(inputs)

                box_preds, label_preds, score_preds = self.box_coder.decode(
                    loc_preds.cpu().data.squeeze(),
                    F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
                    score_thresh=0.6,
                    nms_thresh=0.45)
                pred_boxes.append(box_preds)
                pred_labels.append(label_preds)
                pred_scores.append(score_preds)

                print('debug')
