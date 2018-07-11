import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match,refine_match, log_sum_exp, decode


class RefineMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target,object_score = 0, opt=None):
        super(RefineMultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.object_score = object_score
        self.variance = [0.1, 0.2]
        self.opt = opt

    def forward(self, odm_data, priors, loc_targets, cls_targets, arm_data=None, filter_object=False):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
            arm_data (tuple): arm branch containg arm_loc and arm_conf
            filter_object: whether filter out the  prediction according to the arm conf score
        """

        loc_data, conf_data = odm_data
        if arm_data:
            arm_loc, arm_conf = arm_data

        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.Tensor(num, num_priors)
        for idx in range(num):
            truths = loc_targets[idx]
            labels = cls_targets[idx] + 1  # background as 0

            truths = truths.to(self.opt.device)
            labels = labels.to(self.opt.device)

            # for object detection
            if self.num_classes == 2:
                labels = labels > 0
            if arm_data:
                refine_match(self.threshold, truths, priors, self.variance, labels, loc_t, conf_t, idx, arm_loc[idx])
            else:
                match(self.threshold, truths, priors, self.variance, labels, loc_t, conf_t, idx)

        # wrap targets
        loc_t = loc_t
        conf_t = conf_t
        if arm_data and filter_object:
            arm_conf_data = arm_conf[:,:,1]
            pos = conf_t > 0
            object_score_index = arm_conf_data <= self.object_score
            pos[object_score_index] = 0

        else:
            pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loc_t = loc_t.detach()
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1).long())

        # Hard Negative Mining
        loss_c[pos.view(-1).long()] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1).detach()
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted.long(), size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.sum().item()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c