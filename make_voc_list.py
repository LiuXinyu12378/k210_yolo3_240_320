import os
import re
import numpy as np
import sys
import argparse
import skimage
import skimage.io


def main(train_file: str, output_file: str):
    image_path_list = np.loadtxt(train_file, dtype=str)

    if not os.path.exists('data'):
        os.makedirs('data')

    ann_list = list(image_path_list)
    ann_list = [re.sub(r'JPEGImages', 'labels', s) for s in ann_list]
    ann_list = [re.sub(r'.jpg', '.txt', s) for s in ann_list]

    list_len = len(ann_list)   #坑死
    print(list_len)
    new_list = []
    for i in range(list_len):
        try:
            new_list.append(
                np.array([
                image_path_list[i],
                np.loadtxt(ann_list[i], dtype=float, ndmin=2),
                np.array(skimage.io.imread(image_path_list[i]).shape[0:2])]
            ))
        except:
            pass
    lines = np.array(new_list)

    # lines = np.array([
    #     np.array([
    #         image_path_list[i],
    #         np.loadtxt(ann_list[i], dtype=float, ndmin=2),
    #         np.array(skimage.io.imread(image_path_list[i]).shape[0:2])]
    #     ) for i in range(list_len)])
    #
    np.save(output_file, lines)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',default="new_data.txt", type=str, help='trian.txt file path')
    parser.add_argument('--output_file',default="data/voc_img_ann.npy", type=str, help='output file path')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args.train_file, args.output_file)
