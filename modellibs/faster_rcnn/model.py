import torch
import torch.nn as nn
import torch.nn.functional as F

from modellibs.faster_rcnn.backbones.resnet import ResNet
from modellibs.faster_rcnn.backbones.resnet_fpn import ResNetFPN
from modellibs.faster_rcnn.proposal.rpn import RPN

class FasterRCNN(nn.Module):

    def __init__(self):
        super(FasterRCNN, self).__init__()

        self.backbone = ResNetFPN(resnet_layer=101, pretrained=True)

        self.rpn = RPN()