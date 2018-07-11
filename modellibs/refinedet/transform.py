import random
import numpy as np
import torch
import torchvision.transforms as transforms
from transforms.resize import resize
from transforms.random_distort import random_distort
from transforms.random_flip import random_flip
from modellibs.ssd.box_coder import SSDBoxCoder


class Augmentation(object):

    def __init__(self, opt, subset):
        self.opt = opt
        self.subset = subset

    def __call__(self, img, boxes, labels):
        if self.subset == 'train' or self.subset == 'trainval':
            return self.transform_train(img, boxes, labels)
        else:
            return self.transform_valid(img, boxes, labels)

    def detection_collate(self, batch):
        loc_label = []
        cls_label = []
        imgs = []
        for _, sample in enumerate(batch):
            for _, tup in enumerate(sample):
                if torch.is_tensor(tup):
                    if tup.size()[1:] == torch.Size([4]):
                        loc_label.append(tup)
                    elif len(tup.size()) == 1:
                        cls_label.append(tup)
                    else:
                        imgs.append(tup)

        return (torch.stack(imgs, 0), loc_label, cls_label)

    def transform_train(self, img, boxes, labels):
        img = random_distort(img)
        # if random.random() < 0.5:
        #     img, boxes = random_paste.random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
        # img, boxes, labels = random_crop.random_crop(img, boxes, labels)
        img, boxes = resize(img, boxes, size=(self.opt.img_size, self.opt.img_size), random_interpolation=True)
        img, boxes = random_flip(img, boxes)
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])(img)

        return img, boxes, labels

    def transform_valid(self, img, boxes, labels):
        img, boxes = resize.resize(img, boxes, size=(self.opt.img_size, self.opt.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])(img)
        return img, boxes, labels