# -*- coding: utf-8 -*-
''' 
The order of LF image files may be different with this file.
(Top to Bottom, Left to Right, and so on..)

If you use different LF images, 

you should change our 'func_makeinput.py' file.

# Light field images: input_Cam000-080.png
# All viewpoints = 9x9(81)

# -- LF viewpoint ordering --
# 00 01 02 03 04 05 06 07 08
# 09 10 11 12 13 14 15 16 17
# 18 19 20 21 22 23 24 25 26
# 27 28 29 30 31 32 33 34 35
# 36 37 38 39 40 41 42 43 44
# 45 46 47 48 49 50 51 52 53
# 54 55 56 57 58 59 60 61 62
# 63 64 65 66 67 68 69 70 71
# 72 73 74 75 76 77 78 79 80

'''

import numpy as np
import os
import time
from LF_func.func_pfm import write_pfm, read_pfm
from LF_func.func_makeinput import make_epiinput
from LF_func.func_makeinput import make_input
from LF_func.func_model_cas_s1_025_s2_0125_9_sample_backward_mask import define_cas_LF

import matplotlib.pyplot as plt
import cv2
import imageio

if __name__ == '__main__':

    # Input : input_Cam000-080.png
    # Depth output : image_name.pfm

    dir_output = 'submission_occcas'

    if not os.path.exists(dir_output):
        os.makedirs(dir_output + '/disp_maps')
        os.makedirs(dir_output + '/runtimes')

    # GPU setting ( rtx 3090 - gpu0 )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    '''
    /// Setting 1. LF Images Directory

    LFdir = 'synthetic': Test synthetic LF images (from 4D Light Field Benchmark)
                                   "A Dataset and Evaluation Methodology for 
                                   Depth Estimation on 4D Light Fields".
                                   http://hci-lightfield.iwr.uni-heidelberg.de/

    '''
    LFdir = 'synthetic'

    if (LFdir == 'synthetic'):
        # dir_LFimages = [
        #     'hci_dataset/stratified/backgammon', 'hci_dataset/stratified/dots',
        #     'hci_dataset/stratified/pyramids',
        #     'hci_dataset/stratified/stripes', 'hci_dataset/training/boxes',
        #     'hci_dataset/training/cotton', 'hci_dataset/training/dino',
        #     'hci_dataset/training/sideboard'
        # ]

        dir_LFimages = [
            'hci_dataset/stratified/backgammon', 'hci_dataset/stratified/dots',
            'hci_dataset/stratified/pyramids',
            'hci_dataset/stratified/stripes', 'hci_dataset/training/boxes',
            'hci_dataset/training/cotton', 'hci_dataset/training/dino',
            'hci_dataset/training/sideboard', 'hci_dataset/test/bedroom',
            'hci_dataset/test/bicycle', 'hci_dataset/test/herbs',
            'hci_dataset/test/origami'
        ]

        image_w = 512
        image_h = 512

    # number of views ( 0~8 for 9x9 )
    AngualrViews = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    path_weight = 'CasLF_checkpoint/CasLF_s1_0.25_s2_0.125_9_sample_backward_mask_scratch_lr_ckp/iter0116_s2_valmse0.762_bp1.733.hdf5'
    # path_weight = './pretrain_model_9x9.hdf5'

    img_scale = 0.5
    crop = True

    img_scale_inv = int(1 / img_scale)
    ''' Define Model ( set parameters )'''

    model_learning_rate = 0.001
    model_512 = define_cas_LF(round(img_scale * image_h),
                              round(img_scale * image_w), AngualrViews,
                              model_learning_rate)
    ''' Model Initialization '''

    model_512.load_weights(path_weight)
    dum_sz = model_512.input_shape[0]
    dum = np.zeros((1, dum_sz[1], dum_sz[2], dum_sz[3]), dtype=np.float32)
    tmp_list = []
    for i in range(81):
        tmp_list.append(dum)
    dummy = model_512.predict(tmp_list, batch_size=1)

    avg_attention = []
    time_list = []
    """  Depth Estimation  """
    for image_path in dir_LFimages:
        val_list = make_input(image_path, image_h, image_w, AngualrViews)

        start = time.time()

        # predict
        # val_output_tmp, attention_tmp = model_512.predict(val_list,
        #                                                   batch_size=1)
        if crop:
            crop_size = 256
            stride = 128
            val_output_tmp = np.zeros((1, image_h, image_w), dtype=np.float32)
            val_mask_weight = np.zeros((1, image_h, image_w), dtype=np.float32)
            for i in range(9):
                (row, colume) = divmod(i, 3)
                start_h = row * stride
                start_w = colume * stride
                crop_val_list = [
                    val[:, start_h:start_h + crop_size,
                        start_w:start_w + crop_size, :] for val in val_list
                ]
                val_crop_output_tmp, val_crop_output_tmp_2 = model_512.predict(
                    crop_val_list, batch_size=1)
                val_output_tmp[0, start_h:start_h + crop_size,
                               start_w:start_w +
                               crop_size] += val_crop_output_tmp_2[0, :, :]
                val_mask_weight[0, start_h:start_h + crop_size,
                                start_w:start_w + crop_size] += 1
            val_output_tmp = val_output_tmp / val_mask_weight
        else:
            val_output_tmp, _ = model_512.predict(val_list, batch_size=1)

        runtime = time.time() - start
        time_list.append(runtime)
        # plt.imshow(val_output_tmp[0, :, :])
        print("runtime: %.5f(s)" % runtime)

        with open(dir_output + '/runtimes/%s.txt' % image_path.split('/')[-1],
                  'w') as f:
            f.write(str('%.5f' % runtime))

        # save .pfm file
        imageio.imsave(dir_output + '/%s.jpg' % (image_path.split('/')[-1]),
                       val_output_tmp[0, :, :])
        write_pfm(
            val_output_tmp[0, :, :],
            dir_output + '/disp_maps/%s.pfm' % (image_path.split('/')[-1]))
        print('pfm file saved in %s/%s.pfm' %
              (dir_output, image_path.split('/')[-1]))
    print("average runtime: %.5f(s)" % np.mean(runtime))
