import sys
from collections import OrderedDict
import numpy as np
import os
#import ipdb
from optparse import OptionParser
import csv
import logging
import cv2
from enum import Enum
from random import shuffle


known_classes = []
one_based_classes = {}
zero_based_classes = {}


def format_for_yolo(img_data, classes_dict, base_path='', single_class=True):

    #class_balance = [0] * max(known_classes) # if classes are 1-based there will be a spare element at [0]

    line = "{}".format(os.path.join(base_path, img_data['img_name']))

    for box in img_data['boxes']:
        x1, x2 = box['x_min'], box['x_max']
        y1, y2 = box['y_min'], box['y_max']
        if single_class:
            cls = '0'
        else:
            cls = box['class']
            cls = classes_dict[cls]

#        class_balance[int(cls)] = class_balance[int(cls)] + 1

        box = " {},{},{},{},{}".format(x1, y1, x2, y2, cls)
        line = line + box

    line += '\n'
    return line

def process_file(in_file):
    """
    Input :
    DSCF1013.JPG:[1217,1690,489,201,1],[1774,1619,475,224,2],[2313,1566,460,228,3],[1284,1832,497,231,4],[1879,1798,486,228,5],[2429,1742,475,228,6]
    DSCF1015.JPG:[641,1342,1181,892,3],[2053,1022,1122,735,6]
    DSCF1016.JPG:[1067,1843,1114,613,4],[1954,1278,1021,561,6],[2392,717,964,635,3]
    DSCF1017.JPG:[1834,698,789,422,4]
    """
    file_dataset = []

    with open(in_file, 'r') as inf:
        for line in inf:
            line = line.rstrip('\r\n')
            img_name, bboxes = line.split(':')

            bboxes = bboxes.replace('[', '')
            bboxes = bboxes.replace(']', '')
            tokens = bboxes.split(',')

            entry = {
                 'img_name': img_name,
                 'boxes': [],
                }

            counter = 0
            while counter < len(tokens):
                x_min, y_min, width, height, cls = tokens[counter:counter + 5]
                logging.debug("img={},xmin={},ymin={},width={},height={},clas={}".format(img_name,x_min,y_min,width,height, cls))
                entry['boxes'].append(
                    {
                        'img_name': img_name,
                        'x_min': x_min,
                        'y_min': y_min,
                        'width': width,
                        'height': height,
                        'x_max': str(int(x_min) + int(width)),
                        'y_max': str(int(y_min) + int(height)),
                        'class': cls,
                    })

                if cls not in known_classes:
                    known_classes.append(cls)

                counter += 5

            file_dataset.append(entry)

    return file_dataset
# End of process_file()


def generate_output_file(outf, db, zero_based=True, base_path='', format_list=['yolo'], single_class=False):
    suffix_list = []
    if zero_based:
        cls_dict = zero_based_classes
        suffix_list.append('_zero_based')
    else:
        cls_dict = one_based_classes
        suffix_list.append('_one_based')

    if single_class:
        suffix_list.append('_single_class')

    for fmt in format_list:
        suffix_list.append('_' + fmt)
        out_name = ''.join([outf, *suffix_list, '.txt'])
        print("Writing to : {}".format(out_name))
        with open(out_name, 'w') as of:
            for idx, entry in enumerate(db):
                if fmt == 'yolo':
                    line = format_for_yolo(entry, cls_dict, base_path, single_class)
                    of.write(line)


def gen_image_list(outf, db, base_path=''):
    outf += '.txt'
    with open(outf, 'w') as of:
        print("Generating images list {}".format(outf))
        for img in db:
            of.write(os.path.join(base_path, img['img_name']) + '\n')
# End


def main():
    parser = OptionParser()
    parser.add_option("-i", "--in", dest="input")
    parser.add_option("-o", "--out", dest="out_file")
    parser.add_option("-t", "--train", dest="t", help="desired number of train images", default=-1)
    parser.add_option("-b", "--base", dest="base_path", help="images_base_path", default='')
    (options, args) = parser.parse_args()

    """
    options.input = 'annotationsTrain.txt'
    options.out_file = 'OUT'
    options.base_path = 'home'
    """

    in_dataset = process_file(options.input)
    print("Number of images in dataset : {}".format(len(in_dataset)))

    global zero_based_classes, one_based_classes
    if '0' in known_classes or 0 in known_classes:
        print("Classes/Labels are zero-based. Labels in input file are: {}".format(known_classes))
        zero_based_classes = {k: k for k in known_classes}
        one_based_classes = {k: str(int(k)+1) for k in known_classes}
    else:
        print("Classes/Labels are one-based. Labels in input file are: {}".format(known_classes))
        one_based_classes = {k: k for k in known_classes}
        zero_based_classes = {k: str(int(k) - 1) for k in known_classes}

    gen_test = False
    t = int(options.t)
    if t == -1:
        t = len(in_dataset)
    else:
        assert t < len(in_dataset)
        gen_test = True

    print("Generating Training")
    generate_output_file(options.out_file + '_train', in_dataset[:t], True, options.base_path, ['yolo'], single_class=False)
    generate_output_file(options.out_file + '_train', in_dataset[:t], True, options.base_path, ['yolo'], single_class=True)
    generate_output_file(options.out_file + '_train', in_dataset[:t], False, options.base_path, ['yolo'], single_class=False)
    generate_output_file(options.out_file + '_train', in_dataset[:t], False, options.base_path, ['yolo'], single_class=True)
    gen_image_list(options.out_file + '_train_list', in_dataset[:t], options.base_path)

    if not gen_test:
        return

    print("Generating Testing")
    generate_output_file(options.out_file + '_test', in_dataset[t:], True, options.base_path, ['yolo'], single_class=False)
    generate_output_file(options.out_file + '_test', in_dataset[t:], True, options.base_path, ['yolo'], single_class=True)
    generate_output_file(options.out_file + '_test', in_dataset[t:], False, options.base_path, ['yolo'], single_class=False)
    generate_output_file(options.out_file + '_test', in_dataset[t:], False, options.base_path, ['yolo'], single_class=True)
    gen_image_list(options.out_file + '_test_list', in_dataset[t:], options.base_path)


if __name__ == "__main__":
    main()
