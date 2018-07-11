import torch
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn
from torch.nn.parameter import Parameter

def init_model(opt):

    model_name = opt.model


    if model_name == 'ssd300':
        from modellibs.ssd.model import SSD300
        assert opt.img_size == 300, 'image size does not match'

        model = SSD300(opt.num_classes)
        if opt.use_pretrained:
            state_dict = load_pretrained_weight_vgg('vgg16')
            #state_dict = torch.load('pretrained/vgg16-397923af.pth')
            model.extractor.load_state_dict(state_dict, strict=False)
            print("successfully load pretrained model from 'pretrained/vgg16-397923af.pth'")

    elif model_name == 'ssd512':
        from modellibs.ssd.model import SSD512
        assert opt.img_size == 512, 'image size does not match'
        model = SSD512(opt.num_classes)

    elif model_name == 'fpnssd':
        from modellibs.fpnssd.model import FPNSSD512
        assert opt.img_size == 512, 'image size must be 512'
        model = FPNSSD512(opt.num_classes)
        if opt.use_pretrained:
            state_dict = load_pretrained_weight_resnet('resnet50')
            model.load_state_dict(state_dict, strict=False)
            print("successfully load pretrained model from 'pretrained/resnet50-19c8e357'")

    elif model_name == 'retinanet':
        from modellibs.retinanet.model import RetinaNet
        assert opt.img_size == 512, 'image size must be 512'
        model = RetinaNet(opt.num_classes)
        if opt.use_pretrained:
            state_dict = load_pretrained_weight_resnet('resnet50')
            model.load_state_dict(state_dict, strict=False)
            print("successfully load pretrained model from 'pretrained/resnet50-19c8e357'")

    elif model_name == 'refinedet':
        from modellibs.refinedet.model import RefineDet
        assert opt.img_size == 320, 'image size must be 320'
        model = RefineDet(opt.img_size, opt.num_classes, True).to(opt.device)

        if opt.use_pretrained:
            pass

    elif model_name == 's3fd':
        from modellibs.s3fd.model import s3fd
        assert opt.img_size == 640, 'image size must be 640'
        model = s3fd()

        if opt.use_pretrained:
            state_dict = load_pretrained_weight_vgg('vgg16')

            local_state_dict = iter(model.state_dict().items())
            for i, (name, param) in enumerate(state_dict.items()):

                local_name, local_param = next(local_state_dict)
                try:
                    local_param.copy_(param)
                except:
                    pass
            print("successfully load pretrained model from 'pretrained/vgg16-397923af.pth'")

    elif model_name == 'resnet':
        if opt.resnet_model == 'resnet18':
            from modellibs.resnet.resnet import resnet18
            model = resnet18(pretrained=opt.use_pretrained, opt=opt).to(opt.device)
        elif opt.resnet_model == 'resnet50':
            from modellibs.resnet.resnet import resnet50
            model = resnet50(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    elif model_name == 'chanet':
        if opt.base_model == 'resnet18':
            from modellibs.chanet.chanet import chanet18
            model = chanet18(pretrained=opt.use_pretrained, opt=opt).to(opt.device)

    else:
        raise ValueError('Not implemented yet')

    if opt.resume:
        model_state_dict = torch.load(opt.resume_path)

        opt.start_epochs = model_state_dict['epoch']
        opt.best_loss = model_state_dict['best_loss']
        # opt.best_mAP = model_state_dict['best_mAP']
        model_state_dict = model_state_dict['model']
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
            else:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))

    if opt.device == 'cuda':
        model = model.to(opt.device)
        model = torch.nn.DataParallel(model, device_ids=opt.num_gpus)
        torch.backends.cudnn.benchmark = True

    return model


def load_pretrained_weight_vgg(pretrained_model):

    model_urls = {
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }

    if pretrained_model.startswith('vgg'):
        state_dict = model_zoo.load_url(url=model_urls[pretrained_model], model_dir='pretrained')

    return state_dict


def load_pretrained_weight_resnet(pretrained_model):

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    if pretrained_model.startswith('resnet'):
        state_dict = model_zoo.load_url(url=model_urls[pretrained_model], model_dir='pretrained')

    return state_dict