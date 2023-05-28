# -*- coding: utf-8 -*-

from __future__ import print_function

from LF_func.func_generate_traindata_noise import generate_traindata_for_train
from LF_func.func_generate_traindata_noise import data_augmentation_for_train
from LF_func.func_generate_traindata_noise import generate_traindata512

from LF_func.func_model_cas_s1_025_s2_0125_9_sample_backward_mask import define_cas_LF
from LF_func.func_pfm import read_pfm
from LF_func.func_savedata import display_current_output
from LF_func.util import load_LFdata

import numpy as np
import matplotlib.pyplot as plt

import h5py
import os
import time
import imageio
import datetime
import threading
import cv2
import random
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from math import pow, floor


def step_decay(epoch):
    # print("epoch:", epoch)
    init_lrate = 0.001
    drop = 0.5
    epochs_drop = 30
    lrate = init_lrate * pow(drop, floor(1 + epoch) // epochs_drop)
    print("lr: %f" % (lrate))
    return lrate


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def save_disparity_jet(disparity, filename):
    max_disp = np.nanmax(disparity[disparity != np.inf])
    min_disp = np.nanmin(disparity[disparity != np.inf])
    disparity = (disparity - min_disp) / (max_disp - min_disp)
    disparity = (disparity * 255.0).astype(np.uint8)
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imwrite(filename, disparity)


if __name__ == '__main__':
    ''' 
    We use fit_generator to train CasLF, 
    so here we defined a generator function.
    '''

    class threadsafe_iter:
        """
        Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
        """

        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def __next__(self):
            with self.lock:
                return self.it.__next__()

    def threadsafe_generator(f):
        """
        A decorator that takes a generator function and makes it thread-safe.
        """

        def g(*a, **kw):
            return threadsafe_iter(f(*a, **kw))

        return g

    @threadsafe_generator
    def myGenerator(traindata_all, traindata_label, input_size, label_size,
                    batch_size, AngualrViews, boolmask_img4, boolmask_img6,
                    boolmask_img15):
        while 1:
            (traindata_batch,
             traindata_label_batchNxN) = generate_traindata_for_train(
                 traindata_all, traindata_label, input_size, label_size,
                 batch_size, AngualrViews, boolmask_img4, boolmask_img6,
                 boolmask_img15)

            (traindata_batch,
             traindata_label_batchNxN) = data_augmentation_for_train(
                 traindata_batch, traindata_label_batchNxN, batch_size)

            traindata_batch_list = []
            for i in range(traindata_batch.shape[3]):
                for j in range(traindata_batch.shape[4]):
                    traindata_batch_list.append(
                        np.expand_dims(traindata_batch[:, :, :, i, j],
                                       axis=-1))

            yield (traindata_batch_list, traindata_label_batchNxN)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ''' 
    GPU setting ( Our setting: gtx 1080ti,  
                               gpu number = 0 ) 
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    seed = 42
    setup_seed(seed)
    print("random seed: ", seed)

    networkname = 'CasLF_s1_0.25_s2_0.125_9_sample_backward_mask_scratch_lr'

    iter00 = 0

    load_weight_is = False

    crop = True

    model_learning_rate = 0.001  # 0.0001
    ''' 
    Define Patch-wise training parameters
    '''
    input_size = 32  # 32  # Input size should be greater than or equal to 23 32
    label_size = 32  # 32 # Since label_size should be greater than or equal to 1
    # number of views ( 0~8 for 9x9 )
    AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    batch_size = 16  # 32
    workers_num = 4  # number of threads

    display_status_ratio = 4000  # 2000
    ''' 
    Define directory for saving checkpoint file & disparity output image
    '''
    CasLF_checkpoints_path = 'CasLF_checkpoint/'
    CasLF_output_path = 'CasLF_output/'

    directory_ckp = CasLF_checkpoints_path + "%s_ckp" % (networkname)
    if not os.path.exists(directory_ckp):
        os.makedirs(directory_ckp)

    if not os.path.exists(CasLF_output_path):
        os.makedirs(CasLF_output_path)
    directory_t = CasLF_output_path + '%s' % (networkname)
    if not os.path.exists(directory_t):
        os.makedirs(directory_t)

    txt_name = CasLF_checkpoints_path + 'lf_%s.txt' % (networkname)
    ''' 
    Load Train data from LF .png files
    '''
    print('Load training data...')
    dir_LFimages = [
        'additional/antinous', 'additional/boardgames', 'additional/dishes',
        'additional/greek', 'additional/kitchen', 'additional/medieval2',
        'additional/museum', 'additional/pens', 'additional/pillows',
        'additional/platonic', 'additional/rosemary', 'additional/table',
        'additional/tomb', 'additional/tower', 'additional/town',
        'additional/vinyl'
    ]

    traindata_all, traindata_label = load_LFdata(dir_LFimages)

    traindata, _ = generate_traindata512(traindata_all, traindata_label,
                                         AngualrViews)

    print('Load training data... Complete')
    '''load invalid regions from training data (ex. reflective region)'''
    boolmask_img4 = imageio.imread(
        'hci_dataset/additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png'
    )
    boolmask_img6 = imageio.imread(
        'hci_dataset/additional_invalid_area/museum/input_Cam040_invalid_ver2.png'
    )
    boolmask_img15 = imageio.imread(
        'hci_dataset/additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png'
    )

    boolmask_img4 = 1.0 * boolmask_img4[:, :, 3] > 0
    boolmask_img6 = 1.0 * boolmask_img6[:, :, 3] > 0
    boolmask_img15 = 1.0 * boolmask_img15[:, :, 3] > 0
    ''' 
    Load Test data from LF .png files
    '''
    print('Load test data...')
    dir_LFimages = [
        'stratified/backgammon', 'stratified/dots', 'stratified/pyramids',
        'stratified/stripes', 'training/boxes', 'training/cotton',
        'training/dino', 'training/sideboard'
    ]

    valdata_all, valdata_label = load_LFdata(dir_LFimages)

    valdata, valdata_label = generate_traindata512(valdata_all, valdata_label,
                                                   AngualrViews)

    print('Load test data... Complete')
    ''' 
    Model for patch-wise training  
    '''
    model = define_cas_LF(input_size, input_size, AngualrViews,
                          model_learning_rate)
    ''' 
    Model for predicting full-size LF images  
    '''
    image_w = 512
    image_h = 512
    if crop:
        model_512 = define_cas_LF(int(image_w / 2), int(image_h / 2),
                                  AngualrViews, model_learning_rate)
    else:
        model_512 = define_cas_LF(image_w, image_h, AngualrViews,
                                  model_learning_rate)
    """ 
    load latest_checkpoint
    """
    if load_weight_is:
        print("load pretrain model!!!")
        model.load_weights(
            'CasLF_checkpoint/CasLF_s1_0.25_s2_0.125_9_sample_backward_mask_scratch_lr_ckp/iter0116_s2_valmse0.762_bp1.733',
            by_name=True)
    """ 
    Write date & time 
    """
    f1 = open(txt_name, 'a')
    now = datetime.datetime.now()
    f1.write('\n' + str(now) + '\n\n')
    f1.close()

    my_generator = myGenerator(traindata_all, traindata_label, input_size,
                               label_size, batch_size, AngualrViews,
                               boolmask_img4, boolmask_img6, boolmask_img15)
    best_bad_pixel = 100.0
    if crop:
        crop_size = 256
        stride = 128
        val_output_tmp = np.zeros((8, image_h, image_w), dtype=np.float32)
        val_mask_weight = np.zeros((8, image_h, image_w), dtype=np.float32)
        for i in range(9):
            (row, colume) = divmod(i, 3)
            start_h = row * stride
            start_w = colume * stride
            crop_valdata = [
                val[:, start_h:start_h + crop_size,
                    start_w:start_w + crop_size, :] for val in valdata
            ]
            val_crop_output_tmp, _ = model_512.predict(crop_valdata,
                                                       batch_size=1)
            val_output_tmp[:, start_h:start_h + crop_size, start_w:start_w +
                           crop_size] += val_crop_output_tmp[:, :, :]
            val_mask_weight[:, start_h:start_h + crop_size,
                            start_w:start_w + crop_size] += 1
        val_output_tmp = val_output_tmp / val_mask_weight
        print("crop test!!!")
    else:
        val_output, _ = model_512.predict(valdata, batch_size=1)
        print("test!!!")
    # val_output_s1, val_output_s2 = model_512.predict(valdata, batch_size=1)
    # print("test!!!")
    for iter02 in range(iter00, 120):
        ''' Patch-wise training... start'''
        t0 = time.time()
        # model.fit()
        lrate = LearningRateScheduler(step_decay)
        model.fit_generator(my_generator,
                            steps_per_epoch=int(display_status_ratio),
                            epochs=iter00 + 1,
                            class_weight=None,
                            max_queue_size=10,
                            initial_epoch=iter00,
                            verbose=1,
                            workers=workers_num,
                            callbacks=[lrate])

        iter00 = iter00 + 1
        ''' Test after N*(display_status_ratio) iteration.'''
        weight_tmp1 = model.get_weights()
        model_512.set_weights(weight_tmp1)
        """ Validation """
        ''' Test after N*(display_status_ratio) iteration.'''
        # print("start test!!!")
        if crop:
            crop_size = 256
            stride = 128
            val_output_tmp_s1 = np.zeros((8, image_h, image_w),
                                         dtype=np.float32)
            val_output_tmp_s2 = np.zeros((8, image_h, image_w),
                                         dtype=np.float32)
            val_mask_weight = np.zeros((8, image_h, image_w), dtype=np.float32)
            for i in range(9):
                (row, colume) = divmod(i, 3)
                start_h = row * stride
                start_w = colume * stride
                crop_valdata = [
                    val[:, start_h:start_h + crop_size,
                        start_w:start_w + crop_size, :] for val in valdata
                ]
                val_output_s1, val_output_s2 = model_512.predict(crop_valdata,
                                                                 batch_size=1)
                val_output_tmp_s1[:, start_h:start_h + crop_size,
                                  start_w:start_w +
                                  crop_size] += val_output_s1[:, :, :]
                val_output_tmp_s2[:, start_h:start_h + crop_size,
                                  start_w:start_w +
                                  crop_size] += val_output_s2[:, :, :]
                val_mask_weight[:, start_h:start_h + crop_size,
                                start_w:start_w + crop_size] += 1
            val_output_s1 = val_output_tmp_s1 / val_mask_weight
            val_output_s2 = val_output_tmp_s2 / val_mask_weight
        else:
            val_output_s1, val_output_s2 = model_512.predict(valdata,
                                                             batch_size=1)
        # val_output_s1, val_output_s2 = model_512.predict(valdata, batch_size=1)
        ''' Save prediction image(disparity map) in 'current_output/' folder '''
        # val_error, val_bp = display_current_output(val_output_s1,
        #                                            valdata_label, iter00,
        #                                            directory_t, 'val')
        # validation_mean_squared_error_x100 = 100 * \
        #     np.average(np.square(val_error))
        # validation_bad_pixel_ratio = 100 * np.average(val_bp)

        val_error, val_bp = display_current_output(val_output_s1,
                                                   valdata_label, iter00,
                                                   directory_t, 'val')
        validation_mean_squared_error_x100 = 100 * \
            np.average(np.square(val_error))
        validation_bad_pixel_ratio = 100 * np.average(val_bp)

        save_path_file_new_s1 = (directory_ckp +
                                 '/iter%04d_s1_valmse%.3f_bp%.3f.hdf5' %
                                 (iter00, validation_mean_squared_error_x100,
                                  validation_bad_pixel_ratio))
        print(save_path_file_new_s1)

        val_error, val_bp = display_current_output(val_output_s2,
                                                   valdata_label, iter00,
                                                   directory_t, 'val')
        validation_mean_squared_error_x100 = 100 * \
            np.average(np.square(val_error))
        validation_bad_pixel_ratio = 100 * np.average(val_bp)

        save_path_file_new_s2 = (directory_ckp +
                                 '/iter%04d_s2_valmse%.3f_bp%.3f.hdf5' %
                                 (iter00, validation_mean_squared_error_x100,
                                  validation_bad_pixel_ratio))
        print(save_path_file_new_s2)
        """
        Save bad pixel & mean squared error
        """
        f1 = open(txt_name, 'a')
        f1.write('.' + save_path_file_new_s1 + '\n')
        f1.write('.' + save_path_file_new_s2 + '\n')
        f1.close()
        t1 = time.time()
        ''' save model weights if it get better results than previous one...'''
        if (validation_bad_pixel_ratio < best_bad_pixel):
            best_bad_pixel = validation_bad_pixel_ratio
            model.save(save_path_file_new_s2)
            print("new record!!!")
        else:
            model.save(save_path_file_new_s2)
