#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: test_segment.py
# @Author: Li Chengxin 
# @Time: 2022/5/2 13:37
import skimage.io
from stardist.models import StarDist2D

from prediction import Segmenter
from skimage import io
import matplotlib.pyplot as plt


def save_rois(rois, roi_save_path=None):
    from stardist import export_imagej_rois
    if roi_save_path:
        export_imagej_rois(roi_save_path, rois)
    else:
        raise ValueError('roi_save_path need to be give, in this function or in the class init')



# model = StarDist2D(None, name='segment_20x_model',
#                    basedir='../saved_20x_segment_model/')

seg = Segmenter()
# seg = Segmenter(segment_model=model)
# img = '/home/zje/CellClassify/test/test.tif'
#
# lb, detail = seg.segment(skimage.io.imread(img))
#
# save_rois(detail['coord'], '/home/zje/CellClassify/test/test2.zip')
#
# plt.imshow(lb, cmap='gray')
# plt.show()