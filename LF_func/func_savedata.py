# -*- coding: utf-8 -*-

import numpy as np
import imageio
import cv2


def display_current_output(train_output,
                           traindata_label,
                           iter00,
                           directory_save,
                           train_val='train'):
    '''
        display current results from CasLF 
        and save results in /current_output
    '''
    sz = len(traindata_label)
    train_output = np.squeeze(train_output)
    if (len(traindata_label.shape) > 3
            and traindata_label.shape[-1] == 9):  # traindata
        pad1_half = int(
            0.5 * (np.size(traindata_label, 1) - np.size(train_output, 1)))
        train_label482 = traindata_label[:, 15:-15, 15:-15, 4, 4]
    else:  # valdata
        pad1_half = int(
            0.5 * (np.size(traindata_label, 1) - np.size(train_output, 1)))
        train_label482 = traindata_label[:, 15:-15, 15:-15]

    train_output482 = train_output[:, 15 - pad1_half:482 + 15 - pad1_half,
                                   15 - pad1_half:482 + 15 - pad1_half]

    train_diff = np.abs(train_output482 - train_label482)
    train_bp = (train_diff >= 0.07)

    train_output482_all = np.zeros((2 * 482, sz * 482), np.uint8)
    train_output482_all[0:482, :] = np.uint8(
        25 *
        np.reshape(np.transpose(train_label482, (1, 0, 2)), (482, sz * 482)) +
        100)
    train_output482_all[482:2 * 482, :] = np.uint8(
        25 *
        np.reshape(np.transpose(train_output482, (1, 0, 2)), (482, sz * 482)) +
        100)

    # imageio.imsave(
    #     directory_save + '/' + train_val + '_iter%05d.jpg' % (iter00),
    #     np.squeeze(train_output482_all))

    return train_diff, train_bp


def save_disparity_jet(disparity, filename):
    max_disp = np.nanmax(disparity[disparity != np.inf])
    min_disp = np.nanmin(disparity[disparity != np.inf])
    disparity = (disparity - min_disp) / (max_disp - min_disp)
    disparity = (disparity * 255.0).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imwrite(filename, disparity)


def display_current_output_urban(train_output,
                                 traindata_label,
                                 iter00,
                                 directory_save,
                                 train_val='train'):
    '''
        display current results from CasLF 
        and save results in /current_output
    '''
    sz = len(traindata_label)
    train_output = np.squeeze(train_output)
    if (len(traindata_label.shape) > 3
            and traindata_label.shape[-1] == 9):  # traindata
        pad1_half = int(
            0.5 * (np.size(traindata_label, 1) - np.size(train_output, 1)))
        train_label482 = traindata_label[:, 15:-15, 15:-15, 4, 4]
    else:  # valdata
        pad1_half = int(
            0.5 * (np.size(traindata_label, 1) - np.size(train_output, 1)))
        train_label482 = traindata_label[:, 15:-15, 15:-15]

    train_output482 = train_output[:, 15 - pad1_half:-15 - pad1_half,
                                   15 - pad1_half:-15 - pad1_half]

    train_diff = np.abs(train_output482 - train_label482)
    train_bp = (train_diff >= 0.07)

    train_output482_all = np.zeros((3 * 450, sz * 610), np.uint8)
    train_output482_all[0:450, :] = np.uint8(
        25 *
        np.reshape(np.transpose(train_label482, (1, 0, 2)), (450, sz * 610)) +
        100)
    train_output482_all[450:2 * 450, :] = np.uint8(
        25 *
        np.reshape(np.transpose(train_output482, (1, 0, 2)), (450, sz * 610)) +
        100)
    train_output482_all[2 * 450:3 * 450, :] = np.uint8(
        255 * np.reshape(np.transpose(train_bp, (1, 0, 2)), (450, sz * 610)))
    # train_output482_all[3 * 450:4 * 450, :] = np.uint8(
    #     255 * np.reshape(np.transpose(train_diff, (1, 0, 2)), (450, sz * 610)))

    # imageio.imsave(
    #     directory_save + '/' + train_val + '_iter%05d.jpg' % (iter00),
    #     np.squeeze(train_output482_all))

    # imageio.imsave(
    #     directory_save + '/' + train_val + '_iter%05d.jpg' % (iter00),
    #     np.squeeze(train_output482_all))

    return train_diff, train_bp