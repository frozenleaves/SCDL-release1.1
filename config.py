#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: config.py
# @Author: Li Chengxin 
# @Time: 2022/4/18 15:44


from __future__ import annotations
import os
from typing import Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


TIMES = 20   # Image magnification


# some training parameters
EPOCHS = 150
BATCH_SIZE = 64
NUM_CLASSES = 3  # cell phase num
image_height = 128
image_width = 128
LEARNING_RATE = 1e-6

imageSize: Tuple[int, int] = (2048, 2048)  # input image size for segmentation

channels = 2  # image channels

train_process_20x_detail_data_savefile = '/home/zje/CellClassify/train_detail_20x_new2.csv'
train_process_60x_detail_data_savefile = '/home/zje/CellClassify/train_detail_60x.csv'



save_model_dir_60x = '/home/zje/CellClassify/saved_models/saved_60x_classify_model/model'
dataset_dir_mcy_60x = '/home/zje/CellClassify/train_dataset/train_data_60x_new/train_mcy'
train_dir_mcy_60x = os.path.join(dataset_dir_mcy_60x, "train")
valid_dir_mcy_60x = os.path.join(dataset_dir_mcy_60x, "valid")
test_dir_mcy_60x = os.path.join(dataset_dir_mcy_60x, "test")

dataset_dir_dic_60x = '/home/zje/CellClassify/train_dataset/train_data_60x_new/train_dic'
train_dir_dic_60x = os.path.join(dataset_dir_dic_60x, "train")
valid_dir_dic_60x = os.path.join(dataset_dir_dic_60x, "valid")
test_dir_dic_60x = os.path.join(dataset_dir_dic_60x, "test")



# save_model_dir_20x = '/home/zje/CellClassify/saved_models/saved_20x_classify_model_exp_bbox_best/model'
save_model_dir_20x = '/home/zje/CellClassify/saved_models/saved_20x_classify_model_new2/model'
save_model_dir_20x_best = '/home/zje/CellClassify/saved_models/saved_20x_classify_model_new2_best/model'
dataset_dir_mcy_20x = '/home/zje/CellClassify/train_dataset/train_data_20x_new2/train_mcy'
train_dir_mcy_20x = os.path.join(dataset_dir_mcy_20x, "train")
valid_dir_mcy_20x = os.path.join(dataset_dir_mcy_20x, "valid")
test_dir_mcy_20x = os.path.join(dataset_dir_mcy_20x, "test")

dataset_dir_dic_20x = '/home/zje/CellClassify/train_dataset/train_data_20x_new2/train_dic'
train_dir_dic_20x = os.path.join(dataset_dir_dic_20x, "train")
valid_dir_dic_20x = os.path.join(dataset_dir_dic_20x, "valid")
test_dir_dic_20x = os.path.join(dataset_dir_dic_20x, "test")

# segmentation model config
model_name = 'segment_60x_model'
model_saved_dir = '/home/zje/CellClassify/saved_models/saved_60x_segment_model/'
tain_dataset = '/home/zje/CellClassify/train_dataset/segment_train_60x/train/images'
train_label = '/home/zje/CellClassify/train_dataset/segment_train_60x/train/masks'
valid_size = 0.1


model_name_20x = 'segment_60x_model'
model_saved_dir_20x = '/home/zje/CellClassify/saved_models/saved_20x_segment_model/'
train_dataset_20x = '/home/zje/CellClassify/train_dataset/segment_train_20x/train/images'
train_label_20x = '/home/zje/CellClassify/train_dataset/segment_train_20x/train/masks'
valid_size_20x = 0.1

# choose a network
# model = "resnet18"
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"
