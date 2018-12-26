import sys
from collections import OrderedDict
import numpy as np
import os
import ipdb
from optparse import OptionParser
import csv
import logging
import cv2
from enum import Enum
from random import shuffle


class LIST_ENUMS(Enum):
    CLS = 0
    X_C = 1
    Y_C = 2
    W = 3
    H = 4
# End


# To be filled from first image
IMG_H = 0
IMG_W = 0


# Classes
classes = []
new_classes = []

# format :
# IMG_20181226_180026.txt:
# 0 0.174875 0.483000 0.223250 0.222667
# 2 0.854375 0.623833 0.225750 0.250333
# cls x_center y_center w h
#
def process_file_single(filename):
    lst = []
    imgname = filename.split('.')[:-1] + '.jpg'
    with open(filename, 'r') as inf:
        for line in inf:
            logging.debug("{}".format(line))
#            cls, x_c, y_c, w, h = line.split()
            data = line.split()
            lst.append({
                'imgname': imgname,
                'data': data,
            })

            if data[LIST_ENUMS.CLS] not in classes:
                classes.append(data[LIST_ENUMS.CLS])

    return list
# End


def process_files(dirpath):
    all_bboxes = []

    for f in os.listdir(dirpath):
        if not f.endswith('.txt'):
            continue

        if not os.path.isfile(str(f.split(',)')[:-1]) + '.jpg'):
            logging.warning("Cant find image file for text file {}".format(f))

        img_bboxes = process_file_single(f)
        all_bboxes.append(img_bboxes)

    return all_bboxes
# End


def format_for_simple_parser(outf, entries, base_path):
    """
    Output :
    /data/imgs/img_001.jpg,837,346,981,456,cow
    /data/imgs/img_002.jpg,215,312,279,391,cat
    """

    if os.path.isfile(outf):
        assert 0

    with open(outf, 'w'):
        for entry in entries:
            X_C, Y_C = entry['data'][LIST_ENUMS.X_C], entry['data'][LIST_ENUMS.Y_C]
            W, H = entry['data'][LIST_ENUMS.W], entry['data'][LIST_ENUMS.H]
            CLS = entry['data'][LIST_ENUMS.CLS]

            X_C, Y_C = X_C * IMG_W, Y_C * IMG_H
            W, H = W * IMG_W, H * IMG_H

            im_path = os.path.join(base_path, entry['imgname'])
            x1, x2 = int(X_C - W/2), int(X_C + W/2)
            y1, y2 = int(Y_C - H/2), int(Y_C + H/2)
            cls = new_classes[int(CLS)]

            line = "{},{},{},{},{},{}".format(im_path, x1, y1, x2, y2, cls)
            logging.debug(line)
            outf.write(line)

    # End


def main():
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="in_path", help="Path to input file.", required=True)
    parser.add_option("-o", "--out", dest="out_file", help="Path to out file (A default is chosen if not given).", required=True)
    parser.add_option("-t", "--test", dest="test_part", help="test images ratio", required=True)
    parser.add_option("-b", "--base", dest="base_path", help="images_base_path", required=True)
    (options, args) = parser.parse_args()

    assert (options.t < 0.5)

    # Obtain first image dims
    images = os.listdir(options.p)
    for imname in images:
        if imname.endswith('.jpg'):
            global IMG_H, IMG_W
            IMG_H, IMG_W = cv2.imread(images).shape[:2]
            break

    all_img_bbox = process_files(options.p)
    shuffle(all_img_bbox)

    # remap classes - this will make class num 0 go away
    global new_classes
    global classes
    if '0' in classes:
        new_classes = [str(k + 1) for k in classes]
    else:
        new_classes = [str(k) for k in classes]

    # Check output folders
    base_train = os.path.join(options.base_path, 'TRAIN')
    base_test = os.path.join(options.base_path, 'TRAIN')

    if any(os.path.isdir(bb) for bb in [base_train, base_test]):
        print("TRAIN or TEST folders already exists!")
        assert 0

    os.mkdir(base_test)
    os.mkdir(base_train)

    # Do work ...
    test_idx = int(options.t * len(all_img_bbox))
    format_for_simple_parser(options.out_file + '.train', all_img_bbox[:test_idx])
    format_for_simple_parser(options.out_file + '.test', all_img_bbox[test_idx:])

