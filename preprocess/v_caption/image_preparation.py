import os
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

JAMO = CHO + JOONG + JONG[1:]
FIRST_HANGUL_UNICODE = 0xAC00  # '가'
LAST_HANGUL_UNICODE = 0xD7A3  # '힣'

FIRST_NUMBER_UNICODE = 0x0030  # '0'
LAST_NUMBER_UNICODE = 0x0039  # '9'

FIRST_ALPHABET_UPPER_UNICODE = 0x0041  # 'A'
LAST_ALPHABET_UPPER_UNICODE = 0x005A  # 'Z'

FIRST_ALPHABET_LOWER_UNICODE = 0x0061  # 'a'
LAST_ALPHABET_LOWER_UNICODE = 0x007A  # 'z'


def is_jamo(letter):
    return letter in JAMO

def is_hangul(phrase): # TODO: need tuning!!
    for letter in phrase:
        code = ord(letter)
        if (code < FIRST_HANGUL_UNICODE or code > LAST_HANGUL_UNICODE) and not is_jamo(letter):
            return False

    return True

def is_alphabet(phrase): # TODO: need tuning!!
    unicode_value = ord(phrase)
    if (unicode_value >= FIRST_ALPHABET_UPPER_UNICODE and unicode_value <= LAST_ALPHABET_UPPER_UNICODE) \
       or (unicode_value >= FIRST_ALPHABET_LOWER_UNICODE and unicode_value <= LAST_ALPHABET_LOWER_UNICODE):
        return True
    else:
        return False

def is_number(phrase): # TODO: need tuning!!
    for letter in phrase:
        code = ord(letter)
        if (code < FIRST_NUMBER_UNICODE or code > LAST_NUMBER_UNICODE):
            return False
    return True

def class_assign(caption):
    if len(caption) == 1:
        if is_hangul(caption):

            if is_jamo(caption):

                if caption in result_dict['hangul_jamo']:
                    result_dict['hangul_jamo'][caption] += 1
                else:
                    result_dict['hangul_jamo'][caption] = 1

                class_num = 10000

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
                JONG_class_num = JONG_reduced.index(last_sung)
                class_num = [CHO_class_num, JOONG_class_num, JONG_class_num, config_class_num]

        elif is_alphabet(caption):

            if caption in result_dict['alphabet']:
                result_dict['alphabet'][caption] += 1
            else:
                result_dict['alphabet'][caption] = 1

            class_num = 6

        elif is_number(caption):

            if caption in result_dict['number']:
                result_dict['number'][caption] += 1
            else:
                result_dict['number'][caption] = 1

            class_num = 7

        else:
            if caption in result_dict['etc']:
                result_dict['etc'][caption] += 1
            else:
                result_dict['etc'][caption] = 1

            class_num = 8

    else:
        if caption in result_dict['non_single']:
            result_dict['non_single'][caption] += 1
        else:
            result_dict['non_single'][caption] = 1

        class_num = 10000

        if caption == '...':
            class_num = 8

    return class_num

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--anno_root", type=str, default="/media/son/Repository2/V.DO/V_Caption")
    parser.add_argument("--valid_set", type=list, default=['000481', '000482', '001293', '001294', '001771', '001772'])
    parser.add_argument("--save_dir", type=str, default='hangul_patch')
    parser.add_argument("--task_name", type=str, default='ocr_demo5')

    args = parser.parse_args()

    anno_root = args.anno_root

    make_dir(os.path.join(args.anno_root, args.save_dir))

    with open(os.path.join(anno_root, '{}.json'.format(args.task_name))) as f:
        anno = json.load(f)

    f_1 = open('patch_train.txt', 'w')
    f_2 = open('patch_val.txt', 'w')

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

    count = 0
    for clip in anno['annotation']['clips']:

        clip_name = clip['clip_name']

        if clip_name in args.valid_set:
            f = f_2  # validation set
        else:
            f = f_1  # training set

        clip_count = 0
        for image in clip['images']:

            image_name = image['filename']
            if clip_name == '000487' and image_name == '0001.jpg':
                continue
            elif clip_name == '000487' and image_name == '0020.jpg':
                continue
            elif clip_name == '000492' and image_name == '0104.jpg':
                continue
            elif clip_name == '000494' and image_name == '0040.jpg':
                continue
            elif clip_name == '000494' and image_name == '0081.jpg':
                continue
            elif clip_name == '001307' and image_name == '0004.jpg':
                continue
            elif clip_name == '001309' and image_name == '0043.jpg':
                continue
            elif clip_name == '001309' and image_name == '0044.jpg':
                continue
            elif clip_name == '001309' and image_name == '0045.jpg':
                continue
            elif clip_name == '001309' and image_name == '0046.jpg':
                continue
            elif clip_name == '001784' and image_name == '0070.jpg':
                continue
            elif clip_name == '002210' and image_name == '0002.jpg':
                continue

            im = cv2.imread(os.path.join(anno_root, args.task_name, clip_name, image_name))

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
                class_num = class_assign(caption)
                if not isinstance(class_num, list) and class_num == 10000:
                    continue

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

                image_name_edited = image_name.split('.')[0]
                image_path = os.path.join(args.save_dir, '{}_{}_{}.jpg'.format(clip_name, image_name_edited, sub_count))
                f.write(image_path)
                cv2.imwrite(os.path.join(args.anno_root, args.save_dir, '{}_{}_{}.jpg'.format(clip_name, image_name_edited, sub_count)), cropped_img)

                for i, cls in enumerate(class_num):
                    cls = str(int(cls))
                    f.write(' ' + cls)

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
                    else:
                        raise NotImplementedError

                f.write('\n')
                sub_count += 1
                count += 1

            print('{} processed'.format(count))

    f.close()
    print('done')