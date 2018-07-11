import random
import torch
import torchvision.transforms as transforms
from transforms import resize, random_crop, random_distort, random_flip, random_paste
from modellibs.s3fd.box_coder import S3FDBoxCoder


class ResnetAugmentation(object):

    def __init__(self, opt, subset):
        self.opt = opt
        self.subset = subset

    def __call__(self, img):
        if self.subset == 'train' or self.subset == 'trainval':
            return self.transform_train(img)
        elif self.subset == 'valid':
            return self.transform_valid(img)
        else:
            return self.transform_test(img)

    def transform_train(self, img):
        img = random_distort.random_distort(img)
        img = transforms.Compose([
            transforms.RandomResizedCrop(self.opt.img_size),
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])(img)

        return img

    def transform_valid(self, img):
        img = transforms.Compose([
            transforms.Resize((self.opt.img_size,self.opt.img_size)),
            # transforms.CenterCrop(self.opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])(img)
        return img

    def transform_test(self, img):
        # img, boxes = resize.resize(img, boxes, size=(self.opt.img_size, self.opt.img_size))
        width, height = img.size[0], img.size[1]
        print('image size : {} x {}'.format(width, height))

        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])(img)
        # boxes, labels = self.box_coder.encode(boxes, labels)
        return img