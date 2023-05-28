from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, Reshape, Conv3D, AveragePooling2D, Lambda, UpSampling2D, UpSampling3D, GlobalAveragePooling3D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, add, multiply
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import time
import tensorflow_addons as tfa
from keras_flops import get_flops


def convbn(input, out_planes, kernel_size, stride, dilation):

    seq = Conv2D(out_planes,
                 kernel_size,
                 stride,
                 'same',
                 dilation_rate=dilation,
                 use_bias=False)(input)
    seq = BatchNormalization()(seq)

    return seq


def convbn_3d(input, out_planes, kernel_size, stride):
    seq = Conv3D(out_planes, kernel_size, stride, 'same',
                 use_bias=False)(input)
    seq = BatchNormalization()(seq)

    return seq


def BasicBlock(input, planes, stride, downsample, dilation):
    conv1 = convbn(input, planes, 3, stride, dilation)
    conv1 = Activation('relu')(conv1)
    conv2 = convbn(conv1, planes, 3, 1, dilation)
    if downsample is not None:
        input = downsample

    conv2 = add([conv2, input])
    return conv2


def _make_layer(input, planes, blocks, stride, dilation):
    inplanes = 4
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = Conv2D(planes, 1, stride, 'same', use_bias=False)(input)
        downsample = BatchNormalization()(downsample)

    layers = BasicBlock(input, planes, stride, downsample, dilation)
    for i in range(1, blocks):
        layers = BasicBlock(layers, planes, 1, None, dilation)

    return layers


def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.compat.v1.image.resize_bilinear(
        x, size, align_corners=True))


def UpSampling3DBilinear(size):

    def UpSampling3DBilinear_(x, size):
        shape = K.shape(x)
        x = K.reshape(x, (shape[0] * shape[1], shape[2], shape[3], shape[4]))
        x = tf.compat.v1.image.resize_bilinear(x, size, align_corners=True)
        x = K.reshape(x, (shape[0], shape[1], size[0], size[1], shape[4]))
        return x

    return Lambda(lambda x: UpSampling3DBilinear_(x, size))


def channel_attention(cost_volume):
    x = GlobalAveragePooling3D()(cost_volume)
    x = Lambda(
        lambda y: K.expand_dims(K.expand_dims(K.expand_dims(y, 1), 1), 1))(x)
    x = Conv3D(120, 1, 1, 'same')(x)  # 170
    x = Activation('relu')(x)
    x = Conv3D(15, 1, 1, 'same')(x)  # [B, 1, 1, 1, 15]
    x = Activation('sigmoid')(x)

    # 15 -> 25
    # 0  1  2  3  4
    #    5  6  7  8
    #       9 10 11
    #         12 13
    #            14
    #
    # 0  1  2  3  4
    # 1  5  6  7  8
    # 2  6  9 10 11
    # 3  7 10 12 13
    # 4  8 11 13 14

    x = Lambda(lambda y: K.concatenate([
        y[:, :, :, :, 0:5], y[:, :, :, :, 1:2], y[:, :, :, :, 5:9],
        y[:, :, :, :, 2:3], y[:, :, :, :, 6:7], y[:, :, :, :, 9:12],
        y[:, :, :, :, 3:4], y[:, :, :, :, 7:8], y[:, :, :, :, 10:11],
        y[:, :, :, :, 12:14], y[:, :, :, :, 4:5], y[:, :, :, :, 8:9],
        y[:, :, :, :, 11:12], y[:, :, :, :, 13:15]
    ],
                                       axis=-1))(x)

    x = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 5, 5)))(x)
    x = Lambda(lambda y: tf.pad(y, [[0, 0], [0, 4], [0, 4]], 'REFLECT'))(x)
    attention = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 1, 1, 1, 81)))(x)
    x = Lambda(lambda y: K.repeat_elements(y, 4, -1))(attention)
    return multiply([x, cost_volume]), attention


def feature_extraction(sz_input, sz_input2):
    i = Input(shape=(sz_input, sz_input2, 1))
    firstconv = convbn(i, 4, 3, 1, 1)
    firstconv = Activation('relu')(firstconv)
    firstconv = convbn(firstconv, 4, 3, 1, 1)
    firstconv = Activation('relu')(firstconv)

    layer1 = _make_layer(firstconv, 4, 2, 1, 1)
    layer2 = _make_layer(layer1, 8, 8, 1, 1)
    layer3 = _make_layer(layer2, 16, 2, 1, 1)
    layer4 = _make_layer(layer3, 16, 2, 1, 2)

    layer4_size = (layer4.get_shape().as_list()[1],
                   layer4.get_shape().as_list()[2])

    branch1 = AveragePooling2D((2, 2), (2, 2), 'same')(layer4)
    branch1 = convbn(branch1, 4, 1, 1, 1)
    branch1 = Activation('relu')(branch1)
    branch1 = UpSampling2DBilinear(layer4_size)(branch1)

    branch2 = AveragePooling2D((4, 4), (4, 4), 'same')(layer4)
    branch2 = convbn(branch2, 4, 1, 1, 1)
    branch2 = Activation('relu')(branch2)
    branch2 = UpSampling2DBilinear(layer4_size)(branch2)

    branch3 = AveragePooling2D((8, 8), (8, 8), 'same')(layer4)
    branch3 = convbn(branch3, 4, 1, 1, 1)
    branch3 = Activation('relu')(branch3)
    branch3 = UpSampling2DBilinear(layer4_size)(branch3)

    branch4 = AveragePooling2D((16, 16), (16, 16), 'same')(layer4)
    branch4 = convbn(branch4, 4, 1, 1, 1)
    branch4 = Activation('relu')(branch4)
    branch4 = UpSampling2DBilinear(layer4_size)(branch4)

    output_feature = concatenate(
        [layer2, layer4, branch4, branch3, branch2, branch1])
    lastconv = convbn(output_feature, 16, 3, 1, 1)
    lastconv = Activation('relu')(lastconv)
    lastconv = Conv2D(4, 1, (1, 1), 'same', use_bias=False)(lastconv)

    model = Model(inputs=[i], outputs=[lastconv])

    return model


def _getCostVolume_s1_(inputs):
    shape = K.shape(inputs[0])
    disparity_costs = []
    disparity_values = np.linspace(-4, 4, 33)
    for d in disparity_values:
        if d == 0:
            tmp_list = []
            for i in range(len(inputs)):
                tmp_list.append(inputs[i])
        else:
            tmp_list = []
            for i in range(len(inputs)):
                (v, u) = divmod(i, 9)
                tensor = tfa.image.translate(inputs[i],
                                             [d * (u - 4), d * (v - 4)],
                                             'BILINEAR')
                tmp_list.append(tensor)

        cost = K.concatenate(tmp_list, axis=3)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1)
    cost_volume = K.reshape(cost_volume,
                            (shape[0], 33, shape[1], shape[2], 4 * 81))
    return cost_volume


def _getCostVolume_s2_(inputs_list):
    inputs = inputs_list[0]
    disp = inputs_list[1]
    mask = inputs_list[2]
    disp = K.expand_dims(disp, -1)

    shape = K.shape(inputs[0])
    disparity_costs = []
    disparity_values = np.linspace(-0.5, 0.5, 9)

    for d in disparity_values:
        disp_sample = disp + d
        tmp_list = []
        for i in range(len(inputs)):
            (v, u) = divmod(i, 9)
            flow = tf.concat([disp_sample * (u - 4), disp_sample * (v - 4)],
                             axis=3)
            # mask_veiw = mask[:, i, :, :]
            # mask_input = inputs[i] * mask_veiw
            warp_feat = tfa.image.dense_image_warp(inputs[i], flow)
            mask_warp_feat = warp_feat * mask[:, i, :, :]
            tmp_list.append(mask_warp_feat)

        cost = K.concatenate(tmp_list, axis=3)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1)
    cost_volume = K.reshape(cost_volume,
                            (shape[0], 9, shape[1], shape[2], 4 * 81))
    return cost_volume


def basic(cost_volume):
    feature = 96  # 1 * 75  # 2*75
    dres0 = convbn_3d(cost_volume, feature, 3, 1)
    dres0 = Activation('relu')(dres0)
    dres0 = convbn_3d(dres0, feature, 3, 1)
    cost0 = Activation('relu')(dres0)

    dres1 = convbn_3d(cost0, feature, 3, 1)
    dres1 = Activation('relu')(dres1)
    dres1 = convbn_3d(dres1, feature, 3, 1)
    cost0 = add([dres1, cost0])

    dres4 = convbn_3d(cost0, feature, 3, 1)
    dres4 = Activation('relu')(dres4)
    dres4 = convbn_3d(dres4, feature, 3, 1)
    cost0 = add([dres4, cost0])

    classify = convbn_3d(cost0, feature, 3, 1)
    classify = Activation('relu')(classify)
    cost = Conv3D(1, 3, 1, 'same', use_bias=False)(classify)

    return cost


def disparityregression_s1(input):
    shape = K.shape(input)
    disparity_values = np.linspace(-4, 4, 33)
    x = K.constant(disparity_values, shape=[33])
    x = K.expand_dims(K.expand_dims(K.expand_dims(x, 0), 0), 0)
    x = tf.tile(x, [shape[0], shape[1], shape[2], 1])
    out = K.sum(multiply([input, x]), -1)
    return out


def disparityregression_s2(input):
    shape = K.shape(input)
    disparity_values = np.linspace(-0.5, 0.5, 9)
    x = K.constant(disparity_values, shape=[9])
    x = K.expand_dims(K.expand_dims(K.expand_dims(x, 0), 0), 0)
    x = tf.tile(x, [shape[0], shape[1], shape[2], 1])
    out = K.sum(multiply([input, x]), -1)
    return out


def warp_feature(inputs_list):
    inputs = inputs_list[0]
    disp = inputs_list[1]
    disp = K.expand_dims(disp, -1)
    warp_feature_list = []
    for i in range(len(inputs)):
        (v, u) = divmod(i, 9)
        flow = tf.concat([disp * (u - 4), disp * (v - 4)], axis=3)
        warp_feat = tfa.image.dense_image_warp(inputs[i], flow)
        warp_feature_list.append(warp_feat)
    return warp_feature_list


def warp_feature_mask(inputs_list):
    inputs = inputs_list[0]
    disp = inputs_list[1]
    mask = inputs_list[2]
    disp = K.expand_dims(disp, -1)
    warp_feature_list = []
    shape = K.shape(inputs[0])
    for i in range(len(inputs)):
        (v, u) = divmod(i, 9)
        flow = tf.concat([disp * (u - 4), disp * (v - 4)], axis=3)
        mask_veiw = mask[:, i, :, :]
        # mask_input = inputs[i] * K.repeat_elements(
        #     mask_veiw, shape[3], axis=-1)
        mask_input = inputs[i] * mask_veiw
        warp_feat = tfa.image.dense_image_warp(mask_input, flow)
        warp_feature_list.append(warp_feat)
    return warp_feature_list


def generate_mask(inputs_list):
    inputs = inputs_list[0]
    disp = inputs_list[1]
    disp = K.expand_dims(disp, -1)
    img_ref = inputs[40]
    img_res = []
    for i in range(len(inputs)):
        (v, u) = divmod(i, 9)
        flow = tf.concat([disp * (u - 4), disp * (v - 4)], axis=3)
        img_warped = tfa.image.dense_image_warp(inputs[i], flow)
        img_res.append(abs((img_warped - img_ref)))
    mask = K.stack(img_res, axis=1)
    mask = K.clip(mask, 0, 1)
    out = (1 - mask)**2  # mask [b,81,h,w,c]
    return out


def define_cas_LF(sz_input, sz_input2, view_n, learning_rate):
    """ 81 inputs"""
    input_list = []
    for i in range(len(view_n) * len(view_n)):
        # print('input '+str(i))
        input_list.append(Input(shape=(sz_input, sz_input2, 1)))
    """ 81 features"""
    feature_extraction_layer = feature_extraction(sz_input, sz_input2)
    feature_list = []
    for i in range(len(view_n) * len(view_n)):
        # print('feature '+str(i))
        feature_list.append(feature_extraction_layer(input_list[i]))
    """ cost volume """
    cv = Lambda(_getCostVolume_s1_)(feature_list)
    """ channel attention """
    cv, attention = channel_attention(cv)
    """ cost volume regression """
    cost = basic(cv)
    cost = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1),
                                                 (0, 2, 3, 1)))(cost)
    pred_score_s1 = Activation('softmax')(cost)

    pred_s1 = Lambda(disparityregression_s1)(pred_score_s1)
    # mask = Lambda(generate_mask)([input_list, pred_s1])
    # wrap_feature_list = Lambda(warp_feature)([feature_list, pred_s1])
    # wrap_feature_list = Lambda(warp_feature_mask)(
    #     [feature_list, pred_s1, mask])
    mask = Lambda(generate_mask)([input_list, pred_s1])
    cv2 = Lambda(_getCostVolume_s2_)([feature_list, pred_s1, mask])
    """ channel attention """
    cv2, attention2 = channel_attention(cv2)
    """ cost volume regression """
    cost2 = basic(cv2)
    cost2 = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1),
                                                  (0, 2, 3, 1)))(cost2)
    pred_score_s2 = Activation('softmax')(cost2)

    pred_s2_res = Lambda(disparityregression_s2)(pred_score_s2)
    pred_s2 = pred_s2_res + pred_s1
    model = Model(inputs=input_list, outputs=[pred_s1, pred_s2])

    model.summary()

    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss=['mae', 'mae'])
    # model.compile(optimizer=opt, loss=['mse', 'mse'])

    return model


if __name__ == '__main__':
    input_size = 256  # Input size should be greater than or equal to 23
    label_size = 256  # Since label_size should be greater than or equal to 1
    # number of views ( 0~8 for 9x9 )
    AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    T1 = time.time()
    model = define_cas_LF(input_size, input_size, AngualrViews, 0.001)
    T2 = time.time()
    print('model load: %s s' % ((T2 - T1)))

    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")
    # input_size = 32  # Input size should be greater than or equal to 23
    # label_size = 32  # Since label_size should be greater than or equal to 1
    # # number of views ( 0~8 for 9x9 )
    # AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # # T1 = time.time()
    # model = define_cas_LF(input_size, input_size, AngualrViews, 0.001)

    # dum_sz = model.input_shape[0]
    # dum = np.zeros((1, dum_sz[1], dum_sz[2], dum_sz[3]), dtype=np.float32)
    # tmp_list = []
    # for i in range(81):
    #     tmp_list.append(dum)
    # pred1 = model.predict(tmp_list, batch_size=1)
    # T2 = time.time()
    # for _ in range(10):
    #     pred1 = model.predict(tmp_list, batch_size=1)
    # T3 = time.time()
    # print("average time is: ", (T3 - T2) / 10)
    # input_size = 32  # Input size should be greater than or equal to 23
    # label_size = 32  # Since label_size should be greater than or equal to 1
    # # number of views ( 0~8 for 9x9 )
    # AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # T1 = time.time()
    # model = define_cas_LF(input_size, input_size, AngualrViews, 0.001)
    # T2 = time.time()
    # print('model load: %s s' % ((T2 - T1)))
    # # input_size = 32  # Input size should be greater than or equal to 23
    # # label_size = 32  # Since label_size should be greater than or equal to 1
    # # # number of views ( 0~8 for 9x9 )
    # # AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # # # T1 = time.time()
    # # model = define_cas_LF(input_size, input_size, AngualrViews, 0.001)

    # dum_sz = model.input_shape[0]
    # dum = np.zeros((1, dum_sz[1], dum_sz[2], dum_sz[3]), dtype=np.float32)
    # tmp_list = []
    # for i in range(81):
    #     tmp_list.append(dum)
    # pred1, pred2 = model.predict(tmp_list, batch_size=1)
    # print(np.shape(pred1))
    # print(np.shape(pred2))
    # T2 = time.time()
    # print('model load: %s s' % ((T2 - T1)))
    # input_shape = (12, 32, 32, 4)
    # # x = tf.random.normal(input_shape)
    # x = tf.ones(input_shape)
    # # tf.eye()
    # # x = tf.ones_like(input_shape)
    # x_list = []
    # for _ in range(81):
    #     x_list.append(x)
    # pred_s1 = tf.ones([12, 32, 32, 1])
    # warp_feature_list = warp_feature(x_list, pred_s1)
    # print(np.shape(warp_feature_list))

    # model.predict()

    # for i in range(81):
    #     (v, u) = divmod(i, 9)
    #     flow = tf.concat([pred_s1 * (u - 4), pred_s1 * (v - 4)], axis=3)
    #     warp_feat = tfa.image.dense_image_warp(x_list[i], flow)
    # cost_volume = _getCostVolume_(x_list)
    # cost_volume2 = _getCostVolume050_(x_list)
    # cost_volume3 = _getCostVolume3_(x_list)
    # print(np.shape(cost_volume))
    # input_shape = (12, input_size, label_size, 1)
    # x = tf.random.normal(input_shape)
    # y_list = []
    # T1 = time.time()
    # for _ in range(81):
    #     y_list.append(convbn(x, 4, 3, 1, 1))
    # print(y_list[0].shape)
    # T2 = time.time()
    # print('conv 2d: %s s' % ((T2 - T1)))

    # input_shape = (12, input_size, label_size, 81, 1)
    # x = tf.random.normal(input_shape)
    # T1 = time.time()
    # y = convbn_3d(
    #     x,
    #     out_planes=4,
    #     kernel_size=(3, 3, 1),
    #     stride=1,
    # )
    # print(y.shape)
    # T2 = time.time()
    # print('conv 3d: %s s' % ((T2 - T1)))
