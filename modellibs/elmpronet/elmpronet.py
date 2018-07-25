import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from torch.nn.parameter import Parameter

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, opt):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # config branch
        self.layer3_config = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_config = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool_config = nn.AvgPool2d(3, stride=1)
        self.fc_config = nn.Linear(512 * block.expansion, opt.config_class_num)

        # syllable branch
        # first
        self.inplanes = 512
        self.layer3_first = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_first = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool_first = nn.AvgPool2d(3, stride=1)
        self.fc_first = nn.Linear(512 * block.expansion, opt.first_class_num)

        # middle
        self.inplanes = 512
        self.layer3_middle = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_middle = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool_middle = nn.AvgPool2d(3, stride=1)
        self.fc_middle = nn.Linear(512 * block.expansion, opt.middle_class_num)

        # last
        self.inplanes = 512
        self.layer3_last = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_last = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool_last = nn.AvgPool2d(3, stride=1)
        self.fc_last = nn.Linear(512 * block.expansion, opt.last_class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 56 x 56

        x = self.layer1(x)  # 56 x 56
        x = self.layer2(x)  # 28 x 28

        x_config = self.layer3_config(x)
        x_first = self.layer3_first(x)
        x_middle = self.layer3_middle(x)
        x_last = self.layer3_last(x)

        x_firstp = x_first*x_config
        x_middlep = x_middle *x_config
        x_lastp = x_last*x_config


        x_config = self.layer4_config(x_config)
        x_firstp= self.layer4_first(x_firstp)
        x_middlep= self.layer4_middle(x_middlep)
        x_lastp= self.layer4_last(x_lastp)

        # x_firstp = x_firstp.expand_as(x_first)
        # x_middlep = x_middlep.expand_as(x_middle)
        # x_lastp = x_lastp.expand_as(x_last)

        x_first=x_firstp
        x_middle=x_middlep
        x_last=x_lastp

        x_config = self.avgpool_config(x_config)
        x_first = self.avgpool_first(x_first)
        x_middle = self.avgpool_middle(x_middle)
        x_last = self.avgpool_last(x_last)

        feat_config = x_config.view(x_config.size(0), -1)
        feat_first = x_first.view(x_first.size(0), -1)
        feat_middle = x_middle.view(x_middle.size(0), -1)
        feat_last = x_last.view(x_last.size(0), -1)


        x_config = self.fc_config(feat_config)
        x_first = self.fc_first(feat_first)
        x_middle = self.fc_middle(feat_middle)
        x_last = self.fc_last(feat_last)

        return x_first, x_middle, x_last, x_config


def resnet18(pretrained, opt):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], opt)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained, opt):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], opt)
    if pretrained:
        model_state_dict = model_zoo.load_url(model_urls['resnet18'])
        own_state = model.state_dict()
        for name, param in model_state_dict.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model