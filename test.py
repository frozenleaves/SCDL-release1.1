#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: test.py
# @Author: Li Chengxin 
# @Time: 2022/4/20 16:47
import skimage.io

import utils
import config
from train import get_model
import tensorflow as tf
import numpy as np
import cv2
from libtiff import TIFF
import os
import shutil

model = get_model()
model.load_weights(filepath=config.save_model_dir_60x)


def predict_phase(images: np.ndarray):
    """
    :param image: 一个包含多张图片的数组或列表，其形状
    :return:
    """
    phaseMap = {0: 'G1/G2', 1: 'M', 2: 'S'}
    img = images
    # img = cv2.resize(img, (128, 128)) / 255.0
    tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    # print(tensor.shape)
    if len(images.shape) < 4:
        tensor = tf.expand_dims(tensor, -1)
    # print(tensor.shape)
    prediction = model(tensor, training=False)
    # print(prediction)
    phases = []
    for i in prediction:
        # print(i)
        phase = np.argwhere(i == np.max(i))[0][0]
        # print(phase)
        print(phaseMap[phase])
        phases.append(phaseMap.get(phase))
    return phases

def copy_file():

    name_src = '/home/zje/CellClassify/train_dataset/train_data_60x_new/train_mcy'
    dic_src = '/home/zje/CellClassify/train_dataset/train_data_60x_new/dic'
    dic_dst = '/home/zje/CellClassify/train_dataset/train_data_60x_new/train_dic'
    train = os.path.join(name_src, 'train')
    valid = os.path.join(name_src, 'valid')
    test = os.path.join(name_src, 'test')
    for fd in os.listdir(valid):
        f = os.path.join(valid, fd)
        files = os.listdir(f)
        file_src = [os.path.join(f.replace('train_mcy', 'dic'), x) for x in files]
        src = [s.replace('valid/', '') for s in file_src]

        file_dst = [os.path.join(f.replace('train_mcy', 'train_dic'), y) for y in files]
        for z in zip(src, file_dst):
            if not os.path.exists(os.path.dirname(z[1])):
                os.makedirs(os.path.dirname(z[1]))
            shutil.copy(z[0], z[1])
            print(f'copy f{z[0]} to f{z[1]}')

def normalize_image(image):
    img = image / 255.0
    img = cv2.resize(img, (config.image_width, config.image_height))
    return img

def readTif(filepath):
    """读取并逐帧返回图像数据"""
    tif = TIFF.open(filepath)
    index = 0
    for img in tif.iter_images():
        filename = os.path.basename(filepath).replace('.tif', '-' + str(index).zfill(4) + '.tif')
        index += 1
        yield img, filename


if __name__ == '__main__':
    # im1_dic = cv2.imread('/home/zje/CellClassify/train_dataset/train_data_60x/input_dic/train/G/54800c2bafa83838f7848b8a39b8a33f.tif', -1)
    #
    # im1_dic = normalize_image(im1_dic)
    # im1_mcy = cv2.imread('/home/zje/CellClassify/train_dataset/train_data_60x/input_mcy/train/G/54800c2bafa83838f7848b8a39b8a33f.tif', -1)
    # im1_mcy = normalize_image(im1_mcy)
    # im1 = np.dstack([im1_dic, im1_mcy])
    # data = np.array([im1, im1])
    # predict_phase(data)
    # x = readTif('/home/zje/CellClassify/predict_data/copy_of_1_xy01_mcy.tif')
    # while True:
    #     try:
    #         print(next(x))
    #     except StopIteration:
    #         break
    # utils.tif2png(img='/home/zje/CellClassify/predict_data/dataset/test_pcna.tif',
    #               png_dir='/home/zje/CellClassify/predict_data/dataset/png/test_pcna_60x')

    # copy_file()
    from prediction import Prediction
    mcy = skimage.io.imread('/home/zje/CellClassify/predict_data/test/mcy/10A_rand-0067.tif')
    dic = skimage.io.imread('/home/zje/CellClassify/predict_data/test/dic/10A_rand-0067.tif')
    ann = '/home/zje/CellClassify/predict_data/test/10A_rand-0067.json'
    p = Prediction(mcy, dic, ann, imagesize=(1200, 1200))
    # for i in p.getCell():
    #     print(i.image_mcy.shape)
    #     print(np.max(i.image_mcy))
    #     break
    image_data = []
    id_data = []
    # for cell in p.getCell():
    #     # __dic = cv2.resize(cell.image_dic, (config.image_width, config.image_height))
    #     # __mcy = cv2.resize(cell.image_mcy, (config.image_width, config.image_height))
    #     data = np.dstack([cell.image_dic, cell.image_mcy])
    #     image_data.append(data)
    #     id_data.append(cell.image_id)
    # images = np.array(image_data)
    # print(p.predict_phase(images))
    print(p.predict('10A_rand-0067-0000.png'))

