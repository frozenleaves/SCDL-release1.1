#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: evaluate.py
# @Author: Li Chengxin 
# @Time: 2022/4/20 17:29
import os

import numpy as np
import tensorflow as tf
import config
from prepare_data import generate_datasets_60x, generate_datasets_20x, get_dataset
import skimage
from skimage import transform, io
from train import get_model
import cv2

def read_img(dic_img_path, mcy_img_path):
    dic_img = skimage.io.imread(dic_img_path)
    mcy_img = skimage.io.imread(mcy_img_path)
    # img = np.dstack([dic_img, mcy_img])
    img = np.dstack([transform.resize(dic_img, (config.image_width, config.image_height)).astype(np.float32),
                    transform.resize(mcy_img, (config.image_width, config.image_height)).astype(np.float32)])
    # img = transform.resize(img, (config.image_width, config.image_height))
    # img = tf.convert_to_tensor(img, dtype=tf.float32)
    return img


if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    # train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets_60x()
    test_dataset, test_count = get_dataset(dataset_root_dir_dic=config.test_dir_dic_20x, dataset_root_dir_mcy=config.test_dir_mcy_20x)
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)
    # print(train_dataset)
    # load the model
    model = get_model()
    model.load_weights(filepath=config.save_model_dir_20x)

    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        # rets = []
        # for i in predictions:
        #     rets.append(np.argwhere(i == np.max(i))[0][0])
        # print(rets)

        test_loss(t_loss)
        test_accuracy(labels, predictions)
    #
    # for test_images, test_labels in test_dataset:
    #     test_step(test_images, test_labels)
    #     print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
    #                                                        test_accuracy.result()))
    #
    # print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))

    dic_root_S = '/home/zje/CellClassify/train_dataset/train_data_20x_new/train_dic/test/S/'
    mcy_root_S = '/home/zje/CellClassify/train_dataset/train_data_20x_new/train_mcy/test/S/'
    dic_root_G = '/home/zje/CellClassify/train_dataset/train_data_20x_new/train_dic/test/G/'
    mcy_root_G = '/home/zje/CellClassify/train_dataset/train_data_20x_new/train_mcy/test/G/'
    dic_root_M = '/home/zje/CellClassify/train_dataset/train_data_20x_new/train_dic/test/M/'
    mcy_root_M = '/home/zje/CellClassify/train_dataset/train_data_20x_new/train_mcy/test/M/'
    dic_G = [os.path.join(dic_root_G, x) for x in os.listdir(dic_root_G)]
    mcy_G = [os.path.join(mcy_root_G, x) for x in os.listdir(mcy_root_G)]
    dic_M = [os.path.join(dic_root_M, x) for x in os.listdir(dic_root_M)]
    mcy_M = [os.path.join(mcy_root_M, x) for x in os.listdir(mcy_root_M)]
    dic_S = [os.path.join(dic_root_S, x) for x in os.listdir(dic_root_S)]
    mcy_S = [os.path.join(mcy_root_S, x) for x in os.listdir(mcy_root_S)]
    # d = '/home/zje/CellClassify/train_dataset/train_data_60x/input_dic/train/G/54800c2bafa83838f7848b8a39b8a33f.tif'
    # m = '/home/zje/CellClassify/train_dataset/train_data_60x/input_mcy/train/G/54800c2bafa83838f7848b8a39b8a33f.tif'
    # img, label = read_img(d, m, 'G')
    # predictions = model(np.expand_dims(img, 0), training=False)
    # print(predictions)
    true = 0
    step=1
    predict_img = []
    for i in zip(dic_G, mcy_G):
        predict_img.append(read_img(i[0], i[1]))
        step +=1
        if step >5:
            break
    for i in zip(dic_S, mcy_S):
        predict_img.append(read_img(i[0], i[1]))
        step +=1
        if step >10:
            break
    for i in zip(dic_M, mcy_M):
        predict_img.append(read_img(i[0], i[1]))
        step +=1
        if step >20:
            break
    dt = tf.convert_to_tensor(np.array(predict_img))
    print(dt.shape)
    predictions = model(dt, training=False)
    print(predictions)
    rets = []
    for i in predictions:
        rets.append(np.argwhere(i == np.max(i))[0][0])
    print(rets)
    #     if ret == 1:
    #         true +=1
    #         print(f'\rM accuracy: {true/step} test step: {step}', end='')
    #     step += 1

    # for test_images, test_labels in test_dataset:
    #     print(test_labels)
    #     print(test_images.shape)
    #     predictions = model(test_images, training=False)
    #     rets = []
    #     for i in predictions:
    #         rets.append(np.argwhere(i == np.max(i))[0][0])
    #     print(rets)



