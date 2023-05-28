# -*- coding: utf-8 -*-

import imageio
import numpy as np
import os
import cv2


def make_epiinput(image_path, seq1, image_h, image_w, view_n, RGB):
    traindata_tmp = np.zeros((1, image_h, image_w, len(view_n)),
                             dtype=np.float32)
    i = 0
    if (len(image_path) == 1):
        image_path = image_path[0]

    for seq in seq1:
        tmp = np.float32(
            imageio.imread(image_path + '/input_Cam0%.2d.png' % seq))
        traindata_tmp[0, :, :,
                      i] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] +
                            RGB[2] * tmp[:, :, 2]) / 255
        i += 1
    return traindata_tmp


def make_epiinput_lytro(image_path, seq1, image_h, image_w, view_n, RGB):
    traindata_tmp = np.zeros((1, image_h, image_w, len(view_n)),
                             dtype=np.float32)

    i = 0
    if (len(image_path) == 1):
        image_path = image_path[0]

    for seq in seq1:
        tmp = np.float32(
            imageio.imread(image_path + '/%s_%02d_%02d.png' %
                           (image_path.split("/")[-1], 1 + seq // 9, 1 + seq -
                            (seq // 9) * 9)))
        traindata_tmp[0, :, :,
                      i] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] +
                            RGB[2] * tmp[:, :, 2]) / 255
        i += 1
    return traindata_tmp


def make_epiinput_test(image_path, seq1, image_h, image_w, view_n, RGB):
    traindata_tmp = np.zeros((1, image_h, image_w, len(view_n)),
                             dtype=np.float32)

    i = 0
    if (len(image_path) == 1):
        image_path = image_path[0]

    for seq in seq1:
        tmp_img = imageio.imread(image_path + '/%.2d.png' % (seq + 1))
        tmp_img = cv2.resize(tmp_img, (image_w, image_h))
        tmp = np.float32(tmp_img)
        traindata_tmp[0, :, :,
                      i] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] +
                            RGB[2] * tmp[:, :, 2]) / 255
        i += 1
    return traindata_tmp


def make_epiinput_urban(image_path, seq1, image_h, image_w, view_n, RGB):
    traindata_tmp = np.zeros((1, image_h, image_w, len(view_n)),
                             dtype=np.float32)

    i = 0
    if (len(image_path) == 1):
        image_path = image_path[0]

    for seq in seq1:
        tmp = np.float32(
            imageio.imread(image_path + '/%d_%d.png' % (1 + seq // 9, 1 + seq -
                                                        (seq // 9) * 9)))
        traindata_tmp[0, :, :,
                      i] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] +
                            RGB[2] * tmp[:, :, 2]) / 255
        i += 1

    return traindata_tmp


def make_input(image_path, image_h, image_w, view_n):
    RGB = [0.299, 0.587, 0.114]  ## RGB to Gray // 0.299 0.587 0.114
    '''
    data from http://hci-lightfield.iwr.uni-heidelberg.de/
    Sample images ex: Cam000~ Cam080.png  
    '''

    output_list = []
    for i in range(81):
        if (image_path[:11] == 'hci_dataset'):
            A = make_epiinput(image_path, [i], image_h, image_w, [0], RGB)
        # print(A.shape)
        elif (image_path[:5] == 'lytro'):
            A = make_epiinput_lytro(image_path, [i], image_h, image_w, [0],
                                    RGB)
        elif (image_path[:4] == 'test'):
            A = make_epiinput_test(image_path, [i], image_h, image_w, [0], RGB)
        elif (image_path[:7] == 'UrbanLF'):
            A = make_epiinput_urban(image_path, [i], image_h, image_w, [0],
                                    RGB)
        output_list.append(A)

    return output_list