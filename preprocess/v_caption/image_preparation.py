import os
from os import listdir
from os.path import isfile, join
import argparse
import json
import cv2
import random
import hgtk

"""
Class_Map
0 : 가
1 : 고 
2 : 과
3 : 각
4 : 곡
5 : 곽
6 : ㄱ
7 : a
8 : 1
9 : !
"""

CHO = (
    u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

JOONG = (
    u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅣ',  # 길쭉이 0 ~ 8
    u'ㅗ', u'ㅜ', u'ㅛ', u'ㅠ', u'ㅡ',                             # 넓쩍이 9 ~ 13
    u'ㅘ', u'ㅙ', u'ㅚ', u'ㅝ', u'ㅞ', u'ㅟ', u'ㅢ'                # 길넓이 14 ~ 20
)

JONG = (
    u'', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ', u'ㄺ',
    u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

JONG_reduced = (
    u'', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ',
    u'ㄺ', u'ㄻ', u'ㄼ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

ALPHABET='abcdefghijklnmopqrstuvwxyzABCDEFGHIJKLNMOPQRSTUVWXYZ'

NUMBER='0123456789'

JAMO = CHO + JOONG + JONG[1:]
FIRST_HANGUL_UNICODE = 0xAC00  # '가'
LAST_HANGUL_UNICODE = 0xD7A3  # '힣'

FIRST_NUMBER_UNICODE = 0x0030  # '0'
LAST_NUMBER_UNICODE = 0x0039  # '9'

FIRST_ALPHABET_UPPER_UNICODE = 0x0041  # 'A'
LAST_ALPHABET_UPPER_UNICODE = 0x005A  # 'Z'

FIRST_ALPHABET_LOWER_UNICODE = 0x0061  # 'a'
LAST_ALPHABET_LOWER_UNICODE = 0x007A  # 'z'

Ix_to_title={0:'hangul', 1:'alphabet', 2:'number', 3:'bghangul', 4:'bgalphabet', 5:'bgnumber'}


def is_jamo(letter):
    return letter in JAMO

def is_hangul(phrase): # TODO: need tuning!!
    for letter in phrase:
        code = ord(letter)
        if is_jamo(letter) or (FIRST_HANGUL_UNICODE<=code<=LAST_HANGUL_UNICODE):
            return True
    return False

def is_alphabet(phrase): # TODO: need tuning!!
    unicode_value = ord(phrase)
    if (FIRST_ALPHABET_UPPER_UNICODE<=unicode_value <= LAST_ALPHABET_UPPER_UNICODE) \
       or (FIRST_ALPHABET_LOWER_UNICODE <= unicode_value <= LAST_ALPHABET_LOWER_UNICODE):
        return True
    else:
        return False

def is_number(phrase): # TODO: need tuning!!
    for letter in phrase:
        code = ord(letter)
        if (FIRST_NUMBER_UNICODE <=code <= LAST_NUMBER_UNICODE):
            return True
    return False



def class_assign(caption):

    if len(caption) == 1:
        if is_hangul(caption):

            if is_jamo(caption):

                if caption in result_dict['hangul_jamo']:
                    result_dict['hangul_jamo'][caption] += 1
                else:
                    result_dict['hangul_jamo'][caption] = 1
                class_num = -1

            else:

                if caption in result_dict['hangul']:
                    result_dict['hangul'][caption] += 1
                else:
                    result_dict['hangul'][caption] = 1

                eumso_list = hgtk.letter.decompose(caption)
                first_sung = eumso_list[0]
                middle_sung = eumso_list[1]
                last_sung = eumso_list[2]

                if last_sung == JONG[0]:
                    if middle_sung in JOONG[:9]:
                        config_class_num = 0
                    elif middle_sung in JOONG[9:14]:
                        config_class_num = 1
                    elif middle_sung in JOONG[14:]:
                        config_class_num = 2
                elif last_sung in JONG[1:]:
                    if middle_sung in JOONG[0:9]:
                        config_class_num = 3
                    elif middle_sung in JOONG[9:14]:
                        config_class_num = 4
                    elif middle_sung in JOONG[14:]:
                        config_class_num = 5
                else:
                    raise ValueError

                CHO_class_num = CHO.index(first_sung)
                JOONG_class_num = JOONG.index(middle_sung)
                JONG_class_num = JONG.index(last_sung)
                class_num = [CHO_class_num, JOONG_class_num, JONG_class_num, config_class_num]

        elif is_alphabet(caption):

            if caption in result_dict['alphabet']:
                result_dict['alphabet'][caption] += 1
            else:
                result_dict['alphabet'][caption] = 1

            class_num = ALPHABET.index(caption)

        elif is_number(caption):

            if caption in result_dict['number']:
                result_dict['number'][caption] += 1
            else:
                result_dict['number'][caption] = 1

            class_num = NUMBER.index(caption)

        else:
            if caption in result_dict['etc']:
                result_dict['etc'][caption] += 1
            else:
                result_dict['etc'][caption] = 1

            class_num = -1

    else:
        if caption in result_dict['non_single']:
            result_dict['non_single'][caption] += 1
        else:
            result_dict['non_single'][caption] = 1

        class_num = -1

        if caption == '...':
            class_num =-1

    return class_num

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write_txtnimg(f, root, info,  img):

    image_path = os.path.join('/{}/{}_{}_{}.jpg'.format(info[3], info[0], info[1], info[2]))

    cv2.imwrite(os.path.join(root, '{}/{}_{}_{}.jpg').format(info[3],info[0], info[1],info[2]), img)
    f.write(image_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--anno_root", type=str, default="/home/jade/ws/vdotdo")
    parser.add_argument("--valid_set", type=list, default=['000481', '000482', '001293', '001294', '001771', '001772'])
    parser.add_argument("--save_dir", type=str, default='hangul_patch')
    parser.add_argument("--task_name", type=str, default='background_result')

    args = parser.parse_args()

    anno_root = args.anno_root

    result_dict = dict()
    result_dict['hangul'] = dict()
    result_dict['hangul_jamo'] = dict()
    result_dict['number'] = dict()
    result_dict['alphabet'] = dict()
    result_dict['etc'] = dict()
    result_dict['non_single'] = dict()
    stat_dict = dict()
    stat_dict['first'] = dict()
    stat_dict['middle'] = dict()
    stat_dict['last'] = dict()

    anno_dic={}
    Is_bg=""
    json_files = [pos_json for pos_json in os.listdir(args.anno_root) if pos_json.endswith('.json')]


    for file in json_files:
        with open(os.path.join(anno_root, file)) as f:
            anno_dic[file[:-5]]= json.load(f)

    for key, anno in anno_dic.items():
        print('Start ',key)
        count = 0
        if key.startswith('background'):
            Is_bg="bg"
        else : Is_bg=""

        save_dirh='{}hangul_patch'.format(Is_bg)
        save_dira='{}alphabet_patch'.format(Is_bg)
        save_dirn='{}number_patch'.format(Is_bg)

        make_dir(os.path.join(args.anno_root, save_dirh))
        make_dir(os.path.join(args.anno_root, save_dira))
        make_dir(os.path.join(args.anno_root, save_dirn))

        f_h1 = open(anno_root + '/{}hangul_patch_train.txt'.format(Is_bg), 'w')
        f_h2 = open(anno_root + '/{}hangul_patch_val.txt'.format(Is_bg), 'w')
        f_a1 = open(anno_root + '/{}alphabet_patch_train.txt'.format(Is_bg), 'w')
        f_a2 = open(anno_root + '/{}alphabet_patch_val.txt'.format(Is_bg), 'w')
        f_n1 = open(anno_root + '/{}number_patch_train.txt'.format(Is_bg), 'w')
        f_n2 = open(anno_root + '/{}number_patch_val.txt'.format(Is_bg), 'w')

        for clip in anno['annotation']['clips']:

            clip_name = clip['clip_name']

            divider = random.randint(0, 9)
            if divider <= 1:
                f =  [f_h2, f_a2,f_n2] # validation set
            else:
                f = [f_h1, f_a1, f_n1]  # training set
            # if clip_name in args.valid_set:
            #     f =  [f_h2, f_a2,f_n2] # validation set
            # else:
            #     f = [f_h1, f_a1,f_n1] # training set

            clip_count = 0
            for image in clip['images']:

                image_name = image['filename']
                im = cv2.imread(os.path.join(anno_root, key, clip_name, image_name))
                image_name_edited = image_name.split('.')[0]

                sub_count = 1
                for bbox in image['bbox']:

                    x1 = int(bbox['start_x'])
                    x2 = int(bbox['end_x'])
                    y1 = int(bbox['start_y'])
                    y2 = int(bbox['end_y'])

                    cropped_img = im[y1:y2, x1:x2]

                    # cv2.imshow('photo', cropped_img)
                    # cv2.waitKey(0)

                    # class number assign with caption
                    caption = bbox['caption']
                    if caption is None:
                        continue
                    class_num = class_assign(caption)

                    # Nullify logo
                    if int(clip_name) < 500 and (x2 < 146 and y2 < 120):
                        if clip_count > 14:
                            continue
                        else:
                            clip_count += 1
                    elif int(clip_name) > 500 and int(clip_name) < 1500 and (x2 < 235 and y2 < 97):
                        if clip_count > 11:
                            continue
                        else:
                            clip_count += 1
                    elif int(clip_name) > 1500 and (x2 < 189 and y2 < 101):
                        if clip_count > 20:
                            continue
                        else:
                            clip_count += 1

                    if is_hangul(caption) :

                        image_path = os.path.join('/'+save_dirh,
                                                  '{}_{}_{}.jpg'.format(clip_name, image_name_edited, sub_count))

                        cv2.imwrite(os.path.join(anno_root,save_dirh,
                                                 '{}_{}_{}.jpg'.format(clip_name, image_name_edited, sub_count)),
                                    cropped_img)

                        f[0].write(image_path)

                        # hangul cho joong jong

                        if isinstance(class_num,list):
                            for i, cls in enumerate(class_num):
                                cls = str(int(cls))
                                f[0].write(' ' + cls)

                                if i == 0:
                                    try:
                                        stat_dict['first'][cls] += 1
                                    except:
                                        stat_dict['first'][cls] = 1
                                elif i == 1:
                                    try:
                                        stat_dict['middle'][cls] += 1
                                    except:
                                        stat_dict['middle'][cls] = 1
                                elif i == 2:
                                    try:
                                        stat_dict['last'][cls] += 1
                                    except:
                                        stat_dict['last'][cls] = 1

                        f[0].write('\n')
                        sub_count += 1

                    elif class_num!=-1:
                        if is_alphabet(caption):

                            write_txtnimg(f[1],anno_root,[clip_name, image_name_edited, sub_count, save_dira], cropped_img)

                            f[1].write(' '+str(class_num)+'\n')

                        elif is_number(caption):
                            write_txtnimg(f[2], anno_root,[clip_name, image_name_edited, sub_count, save_dirn], cropped_img)
                            f[2].write(' '+str(class_num)+'\n')


                count += 1

                print('{} processed'.format(count))

            print('done')
    for file in f:
        file.close()
