import os
import math
import cv2
import numpy as np

def eclipse_to_rect(obj, height, width):
    maj_rad, min_rad = float(obj[0]), float(obj[1])
    angle, xcenter, ycenter = float(obj[2]), float(obj[3]), float(obj[4])
    cosin = math.cos(math.radians(-angle))
    sin = math.sin(math.radians(-angle))

    x1 = cosin * (-min_rad) - sin * (-maj_rad) + xcenter
    y1 = sin * (-min_rad) + cosin * (-maj_rad) + ycenter
    x2 = cosin * (min_rad) - sin * (-maj_rad) + xcenter
    y2 = sin * (min_rad) + cosin * (-maj_rad) + ycenter
    x3 = cosin * (min_rad) - sin * (maj_rad) + xcenter
    y3 = sin * (min_rad) + cosin * (maj_rad) + ycenter
    x4 = cosin * (-min_rad) - sin * (maj_rad) + xcenter
    y4 = sin * (-min_rad) + cosin * (maj_rad) + ycenter
    wid = [x1, x2, x3, x4]
    hei = [y1, y2, y3, y4]
    xmin_ = int(min(wid))
    xmax_ = int(max(wid))
    ymin_ = int(min(hei))
    ymax_ = int(max(hei))
    if xmin_ < 0: xmin_ = 0
    if ymin_ < 0: ymin_ = 0
    if xmax_ > width: xmax_ = width
    if ymax_ > height: ymax_ = height
    cls_label = '0' if obj[5] == '1' else None

    return [str(xmin_), str(ymin_), str(xmax_), str(ymax_), cls_label]


def parse_fddb_annotation(data_root, annotation_dir, mode, dst_txt_path, valid_fold):

    txtfiles = os.listdir(annotation_dir)
    txtfiles.sort()

    # fold_number = range(0,10) if mode == 'train' else range(8, 10)
    if mode == 'valid':
        fold_number = [valid_fold]
    else:
        fold_number = [i+1 for i in range(0,10) if i+1 != valid_fold]

    anno = []
    log = ''
    for i, txtfile in enumerate(txtfiles):
        if (txtfile.split('-')[-1] == 'ellipseList.txt') and (int(txtfile.split('-')[-2]) in fold_number):
            with open(os.path.join(annotation_dir, txtfile)) as txt:
                lines = txt.readlines()
                for j, line in enumerate(lines):
                    if len(line.split('/')) == 5:
                        anno.append(line.split()[0])
                        img = cv2.imread(os.path.join(data_root, line.split()[0]+'.jpg'))
                        height, width, channels = img.shape
                        num_object = int(lines[j+1])
                        for obj_idx in range(1, 1+num_object):
                            obj = lines[j+1+obj_idx].split()
                            anno += eclipse_to_rect(obj, height, width)
                        log += ' '.join(anno)
                        log += '\n'
                    else:
                        anno = []
                        continue

    with open(dst_txt_path, 'w') as txt:
        txt.write(log)


if __name__ == '__main__':

    mode = ['train', 'val']
    data_root = '/home/myunggi/Project/Data/FDDB/imgs'
    annotation_dir = '/home/myunggi/Project/Data/FDDB/anno'
    dst_root = '../../data/'
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    dst_txt_root = os.path.join(dst_root, 'fddb')
    if not os.path.exists(dst_txt_root):
        os.mkdir(dst_txt_root)

    for m in mode:
        dst_txt_path = os.path.join(dst_txt_root, 'fddb_%s.txt' % m)
        parse_fddb_annotation(data_root, annotation_dir, m, dst_txt_path)
