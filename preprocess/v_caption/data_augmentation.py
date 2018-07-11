import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--anno_root", type=str, default="/media/son/Repository2/V.DO/V_Caption")
    parser.add_argument("--save_dir", type=str, default='hangul_patch')

    args = parser.parse_args()

    anno_root = args.anno_root

    hangul_dir = os.path.join(anno_root, 'phd08')
