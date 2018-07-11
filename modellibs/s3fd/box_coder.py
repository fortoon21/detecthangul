'''Encode object boxes and labels.'''
import math
import torch
import itertools
import torch.nn.functional as F

from utils import meshgrid
from utils.box import box_iou, box_nms, change_box_order


class S3FDBoxCoder:
    def __init__(self, opt):
        self.steps = opt.steps
        self.box_sizes = opt.box_sizes
        self.aspect_ratios = opt.aspect_ratios
        self.fm_sizes = opt.fm_sizes
        self.default_boxes = self._get_default_boxes()

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.fm_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):
                cx = (w + 0.5) * self.steps[i]
                cy = (h + 0.5) * self.steps[i]

                s = self.box_sizes[i]
                boxes.append((cx, cy, s, s))

                # s = math.sqrt(self.box_sizes[i] * self.box_sizes[i+1])
                # boxes.append((cx, cy, s, s))

                s = self.box_sizes[i]
                for ar in self.aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes, labels):
        '''Encode target bounding boxes and class labels.
        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)
        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (tuple) model input size of (w,h).
        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        anchor_boxes = self.default_boxes  # xywh
        anchor_boxes = change_box_order(anchor_boxes, 'xywh2xyxy')
        default_boxes_ = anchor_boxes

        ious = box_iou(anchor_boxes, boxes)
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        boxes = change_box_order(boxes, 'xyxy2xywh')
        anchor_boxes = change_box_order(anchor_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        # variances = (1, 1)
        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:] / variances[0]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:]) / variances[1]
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[max_ids]

        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1                  # mark ignored to -1
        # return loc_targets, cls_targets, self.default_boxes, default_boxes_
        return loc_targets, cls_targets

    def encode_(self, image, boxes, labels):
        '''Encode target bounding boxes and class labels.
        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)
        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (tuple) model input size of (w,h).
        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        anchor_boxes = self.default_boxes  # xywh
        anchor_boxes = change_box_order(anchor_boxes, 'xywh2xyxy')
        default_boxes_ = anchor_boxes

        ious = box_iou(anchor_boxes, boxes)
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        boxes = change_box_order(boxes, 'xyxy2xywh')
        anchor_boxes = change_box_order(anchor_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        # variances = (1, 1)
        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:] / variances[0]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:]) / variances[1]
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[max_ids]

        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1                  # mark ignored to -1
        # return loc_targets, cls_targets, self.default_boxes, default_boxes_
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Decode predicted loc/cls back to real box locations and class labels.
        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.
        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
        variances = (0.1, 0.2)
        # variances = (1, 1)

        xy = loc_preds[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        wh = torch.exp(loc_preds[:, 2:] * variances[1]) * self.default_boxes[:, 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes-1):
            score = cls_preds[:,i+1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            keep = box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        try:
            boxes = torch.cat(boxes, 0)
            labels = torch.cat(labels, 0)
            scores = torch.cat(scores, 0)
        except:
            boxes = None
            labels = None
            scores = None

        return boxes, labels, scores

    def decode_(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Decode predicted loc/cls back to real box locations and class labels.
        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.
        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
        # variances = (0.1, 0.2)
        variances = (1, 1)
        xy = loc_preds[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        wh = torch.exp(loc_preds[:, 2:] * variances[1]) * self.default_boxes[:, 2:]
        box_preds = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = 1
        for i in range(num_classes):
            score = cls_preds  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            keep = box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        try:
            boxes = torch.cat(boxes, 0)
            labels = torch.cat(labels, 0)
            scores = torch.cat(scores, 0)
        except:
            boxes = None
            labels = None
            scores = None

        return boxes, labels, scores

    def decode__(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Decode predicted loc/cls back to real box locations and class labels.
        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.
        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''

        self.steps = (4, 8, 16, 32, 64, 128)
        self.box_sizes = (16, 32, 64, 128, 256, 512)
        self.aspect_ratios = ((), (), (), (), (), ())

        boxes = []
        score = []
        for i in range(len(cls_preds)):
            cls_preds[i] = F.softmax(cls_preds[i], dim=0)
        for i in range(len(loc_preds)):
            oreg, ocls = loc_preds[i].permute(1,2,0).data.cpu(), cls_preds[i].permute(1,2,0).data.cpu()
            # oreg, ocls = loc_preds[i].squeeze(0).permute(1,2,0).data.cpu(), cls_preds[i].squeeze(0).permute(1,2,0).data.cpu()
            FH, FW, score_num = ocls.size() # feature map size
            for Findex in range(FH*FW):
                windex, hindex = Findex % FW, Findex // FW
                cx = (windex + 0.5) * self.steps[i]
                cy = (hindex + 0.5) * self.steps[i]

                if ocls[hindex, windex, 1] > score_thresh:
                    s = self.box_sizes[i]
                    loc = oreg[hindex, windex, :].unsqueeze(0)
                    prior = torch.Tensor([cx, cy, s, s]).unsqueeze(0)

                    variances = (0.1, 0.2)

                    xy = loc[:, :2] * variances[0] * prior[:, 2:] + prior[:, :2]
                    wh = torch.exp(loc[:, 2:] * variances[1]) * prior[:, 2:]
                    boxes.append(torch.cat([xy - wh / 2, xy + wh / 2], 1))
                    score.append(ocls[hindex, windex, 1])

                # if ocls[hindex, windex, 1, 1] > score_thresh:
                #     s = math.sqrt(self.box_sizes[i] * self.box_sizes[i + 1])
                #     loc = oreg[hindex, windex, 1, :].unsqueeze(0)
                #     prior = torch.Tensor([cx, cy, s, s]).unsqueeze(0)
                #
                #     variances = (0.1, 0.2)
                #
                #     xy = loc[:, :2] * variances[0] * prior[:, 2:] + prior[:, :2]
                #     wh = torch.exp(loc[:, 2:] * variances[1]) * prior[:, 2:]
                #     boxes.append(torch.cat([xy - wh / 2, xy + wh / 2], 1))
                #     score.append(ocls[hindex, windex, 1, 1])

                s = self.box_sizes[i]
                for j, ar in enumerate(self.aspect_ratios[i]):
                    if ocls[hindex, windex, 1 + j*2, 1] > score_thresh:
                        loc = oreg[hindex, windex, 1 + j*2, :].unsqueeze(0)
                        prior = torch.Tensor([cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)]).unsqueeze(0)
                        variances = (0.1, 0.2)

                        xy = loc[:, :2] * variances[0] * prior[:, 2:] + prior[:, :2]
                        wh = torch.exp(loc[:, 2:] * variances[1]) * prior[:, 2:]
                        boxes.append(torch.cat([xy - wh / 2, xy + wh / 2], 1))
                        score.append(ocls[hindex, windex, 1 + j*2, 1])

                    if ocls[hindex, windex, 1 + j*2 + 1, 1] > score_thresh:
                        loc = oreg[hindex, windex, 1 + j*2 + 1, :].unsqueeze(0)
                        prior = torch.Tensor([cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)]).unsqueeze(0)
                        variances = (0.1, 0.2)

                        xy = loc[:, :2] * variances[0] * prior[:, 2:] + prior[:, :2]
                        wh = torch.exp(loc[:, 2:] * variances[1]) * prior[:, 2:]
                        boxes.append(torch.cat([xy - wh / 2, xy + wh / 2], 1))
                        score.append(ocls[hindex, windex, 1 + j*2 + 1, 1])

        try:
            box = torch.cat(boxes, 0)
        except:
            boxes = None
            labels = None
            scores = None
            return boxes, labels, scores

        score = torch.Tensor(score)

        boxes = []
        labels = []
        scores = []

        keep = box_nms(box, score, nms_thresh)
        boxes.append(box[keep])
        labels.append(torch.LongTensor(len(box[keep])).fill_(i))
        scores.append(score[keep])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)

        return boxes, labels, scores

    def decode___(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        '''Decode predicted loc/cls back to real box locations and class labels.
        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.
        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''

        self.steps = (4, 8, 16, 32, 64, 128, 256, 512)
        self.box_sizes = (17.92, 35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6)
        self.aspect_ratios = ((), (2,), (2,), (2,), (2,), (2,), (2,), (2,))

        boxes = []
        score = []
        for i in range(len(cls_preds)):
            cls_preds[i] = F.sigmoid(cls_preds[i].squeeze())
        for i in range(len(loc_preds)):
            oreg, ocls = loc_preds[i].squeeze().data.cpu(), cls_preds[i].data.cpu()
            FH, FW, anchor_num = ocls.size() # feature map size
            for Findex in range(FH*FW):
                windex, hindex = Findex % FW, Findex // FW
                cx = (windex + 0.5) * self.steps[i]
                cy = (hindex + 0.5) * self.steps[i]

                if ocls[hindex, windex, 0] > score_thresh:
                    s = self.box_sizes[i]
                    loc = oreg[hindex, windex, 0, :].unsqueeze(0)
                    prior = torch.Tensor([cx, cy, s, s]).unsqueeze(0)

                    variances = (1, 1)

                    xy = loc[:, :2] * variances[0] * prior[:, 2:] + prior[:, :2]
                    wh = torch.exp(loc[:, 2:] * variances[1]) * prior[:, 2:]
                    boxes.append(torch.cat([xy - wh / 2, xy + wh / 2], 1))
                    score.append(ocls[hindex, windex, 0])

                if ocls[hindex, windex, 1] > score_thresh:
                    s = math.sqrt(self.box_sizes[i] * self.box_sizes[i + 1])
                    loc = oreg[hindex, windex, 1, :].unsqueeze(0)
                    prior = torch.Tensor([cx, cy, s, s]).unsqueeze(0)

                    variances = (1, 1)

                    xy = loc[:, :2] * variances[0] * prior[:, 2:] + prior[:, :2]
                    wh = torch.exp(loc[:, 2:] * variances[1]) * prior[:, 2:]
                    boxes.append(torch.cat([xy - wh / 2, xy + wh / 2], 1))
                    score.append(ocls[hindex, windex, 1])

                s = self.box_sizes[i]
                for j, ar in enumerate(self.aspect_ratios[i]):
                    if ocls[hindex, windex, 2 + j*2] > score_thresh:
                        loc = oreg[hindex, windex, 2 + j*2, :].unsqueeze(0)
                        prior = torch.Tensor([cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)]).unsqueeze(0)
                        variances = (1, 1)

                        xy = loc[:, :2] * variances[0] * prior[:, 2:] + prior[:, :2]
                        wh = torch.exp(loc[:, 2:] * variances[1]) * prior[:, 2:]
                        boxes.append(torch.cat([xy - wh / 2, xy + wh / 2], 1))
                        score.append(ocls[hindex, windex, 2 + j*2])

                    if ocls[hindex, windex, 2 + j*2 + 1] > score_thresh:
                        loc = oreg[hindex, windex, 2 + j*2 + 1, :].unsqueeze(0)
                        prior = torch.Tensor([cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)]).unsqueeze(0)
                        variances = (1, 1)

                        xy = loc[:, :2] * variances[0] * prior[:, 2:] + prior[:, :2]
                        wh = torch.exp(loc[:, 2:] * variances[1]) * prior[:, 2:]
                        boxes.append(torch.cat([xy - wh / 2, xy + wh / 2], 1))
                        score.append(ocls[hindex, windex, 2 + j*2 + 1])

        try:
            box = torch.cat(boxes, 0)
        except:
            boxes = None
            labels = None
            scores = None
            return boxes, labels, scores

        score = torch.Tensor(score)

        boxes = []
        labels = []
        scores = []

        keep = box_nms(box, score, nms_thresh)
        boxes.append(box[keep])
        labels.append(torch.LongTensor(len(box[keep])).fill_(i))
        scores.append(score[keep])

        boxes = torch.cat(boxes, 0)
        labels = torch.cat(labels, 0)
        scores = torch.cat(scores, 0)

        return boxes, labels, scores