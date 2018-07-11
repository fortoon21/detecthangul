import random
import torch
import torchvision.transforms as transforms
from transforms import resize, random_crop, random_distort, random_flip, random_paste
from modellibs.ssd.box_coder import SSDBoxCoder


class SSDAugmentation(object):

    def __init__(self, opt, subset):
        self.opt = opt
        self.subset = subset
        self.box_coder = SSDBoxCoder(opt)

    def __call__(self, img, boxes, labels):
        if self.subset == 'train' or self.subset == 'trainval':
            return self.transform_train(img, boxes, labels)
        else:
            return self.transform_valid(img, boxes, labels)

    def detection_collate(self, batch):
        return None

    def transform_train(self, img, boxes, labels):
        img = random_distort.random_distort(img)
        if random.random() < 0.5:
            img, boxes = random_paste.random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
        img, boxes, labels = random_crop.random_crop(img, boxes, labels)
        img, boxes = resize.resize(img, boxes, size=(self.opt.img_size, self.opt.img_size), random_interpolation=True)
        img, boxes = random_flip.random_flip(img, boxes)
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])(img)
        boxes, labels = self.box_coder.encode(boxes, labels)

        return img, boxes, labels

    def transform_valid(self, img, boxes, labels):
        img, boxes = resize.resize(img, boxes, size=(self.opt.img_size, self.opt.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])(img)
        boxes, labels = self.box_coder.encode(boxes, labels)
        return img, boxes, labels