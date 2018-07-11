from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class V_Caption(data.Dataset):
    '''Load image/labels/boxes from a list file.
    The list file is like:
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, root, list_file, transform=None):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            label = []
            for i in range(1, 5):
                c = splited[i]
                label.append(int(c))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.
        Args:
          idx: (int) image index.
        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        labels_first = self.labels[idx][0].clone()
        labels_middle = self.labels[idx][1].clone()
        labels_last = self.labels[idx][2].clone()
        labels_config = self.labels[idx][3].clone()

        if self.transform:
            img = self.transform(img)
        return img, (labels_first, labels_middle, labels_last, labels_config)

    def __len__(self):
        return self.num_imgs