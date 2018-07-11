import os


def parse_wider_annotation(anno_path, dst_txt_path):

    anno = []
    log = ''
    with open(anno_path, 'r') as txt:
        data = txt.readlines()
        for i, line in enumerate(data):
            if line.endswith('.jpg\n'):
                anno.append(line.split()[0])
                num_object = int(data[i+1])
                for obj_idx in range(1, 1+num_object):
                    bbox = data[i + 1 + obj_idx].split()[:4]
                    bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3])]
                    bbox = [str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]
                    anno += bbox
                    anno.append('0')
                log += ' '.join(anno)
                log += '\n'
                print('----debug')
            else:
                anno = []
                continue
    with open(dst_txt_path, 'w') as txt:
        txt.write(log)


if __name__ == '__main__':

    mode = ['train', 'val']
    data_root = '/home/myunggi/Project/Data/WIDER'
    annotation_path = 'wider_face_split/wider_face_%s_bbx_gt.txt'
    dst_root = '../../data/'
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    dst_txt_root = os.path.join(dst_root, 'wider')
    if not os.path.exists(dst_txt_root):
        os.mkdir(dst_txt_root)

    for m in mode:

        anno_path = os.path.join(data_root, annotation_path % m)
        dst_txt_path = os.path.join(dst_txt_root, 'wider_%s.txt' % m)
        parse_wider_annotation(anno_path, dst_txt_path)
