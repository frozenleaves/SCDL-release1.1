"""
分割细胞图像并预测细胞周期
"""

from __future__ import print_function, unicode_literals, absolute_import, division, annotations

import json
import logging
import os
import gc
import time
import skimage

import utils
import cv2
import retry
import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from utils import ConverterXY, extractRoiFromImg, coordinate2mask
from libtiff import TIFF
import skimage.exposure as exposure
from skimage.util import img_as_ubyte
import config
from train import get_model
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)

PHASE_PREDICT_MODEL = None
STARDIST_MODEL = None
MODEL_USING_INDEX = 0


def reload_phase_predict_model(times=config.TIMES):
    global PHASE_PREDICT_MODEL
    if MODEL_USING_INDEX % 20 == 0:
        del PHASE_PREDICT_MODEL
        gc.collect()
        PHASE_PREDICT_MODEL = get_model()
        if times == 60:
            PHASE_PREDICT_MODEL.load_weights(filepath=config.save_model_dir_60x)
        elif times == 20:
            PHASE_PREDICT_MODEL.load_weights(filepath=config.save_model_dir_20x)
        else:
            raise ValueError(f"Image magnification should be 20 or 60, got {config.TIMES} instead")
        return PHASE_PREDICT_MODEL
    else:
        return PHASE_PREDICT_MODEL


def reload_stardist_model(__model=None):
    global STARDIST_MODEL
    if MODEL_USING_INDEX % 20 == 0:
        del STARDIST_MODEL
        if __model:
            STARDIST_MODEL = __model
        else:
            STARDIST_MODEL = StarDist2D.from_pretrained('2D_versatile_fluo')

        gc.collect()
        return STARDIST_MODEL
    else:
        return STARDIST_MODEL


def readTif(filepath, normalize=False):
    """读取并逐帧返回图像数据"""
    tif = TIFF.open(filepath)
    index = 0
    for img in tif.iter_images():
        filename = os.path.basename(filepath).replace('.tif', '-' + str(index).zfill(4) + '.tif')
        index += 1
        if normalize:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)/255.0
        yield img, filename


class Predictor:
    def __init__(self, times=config.TIMES):
        self.model = get_model()
        if times == 60:
            self.model.load_weights(filepath=config.save_model_dir_60x)
        elif times == 20:
            #self.model.load_weights(filepath=config.save_model_dir_20x_best)
            self.model.load_weights('./saved_classify_models/saved_20x_classify_model_new2/model')
            #self.model.load_weights('../saved_models/saved_20x_classify_model_new/model')
        else:
            raise ValueError(f"Image magnification should be 20 or 60, got {config.TIMES} instead")

    def predict(self, images):
        """
        :param images: 一个包含多张图片的数组或列表，其形状为[image_count, image_width, image_height, image_channels]
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
        prediction = self.model(tensor, training=False)
        # print(prediction)
        phases = []
        for i in prediction:
            # print(i)
            phase = np.argwhere(i == np.max(i))[0][0]
            # print(phase)
            # print(phaseMap[phase])
            phases.append(phaseMap.get(phase))
        return phases


class Prediction:
    def __init__(self, mcy: np.ndarray | str, dic: np.ndarray | str, annotation_json, imagesize=None, predictor=None):
        self.imagesize = imagesize
        self.imgMcy = mcy
        self.imgDic = dic
        self.parser = utils.JsonParser(annotation_json)
        self.images = self.parser.images
        if predictor is None:
            self.predictor = Predictor()
        else:
            self.predictor = predictor

    @staticmethod
    def split(array):
        """返回图像中包含单个细胞的最小外接矩形区域"""
        x0, x1 = utils.find_positions(array)
        y0, y1 = utils.find_positions(array.T)
        return array[x0: x1, y0: y1], np.max([x1 - x0, y1 - y0])

    def getCell(self, cellfilter=20):
        """单细胞图像生成器
        """
        image_data = []
        for img in self.images:
            instances = self.parser.idMap.get(img)
            for instance in instances:
                data = utils.Data()
                coord = instances[instance]
                mask = coordinate2mask([coord], image_size=self.imagesize)[0]
                if type(self.imgMcy) is str:
                    image_mcy = extractRoiFromImg(os.path.join(self.imgMcy, img.replace('.png', '.tif')), mask)
                elif type(self.imgMcy) is np.ndarray:
                    image_mcy = extractRoiFromImg(self.imgMcy, mask)
                else:
                    raise TypeError(
                        f"need image path or image array, but got type {type(self.imgMcy)} in {self.imgMcy}")
                if type(self.imgDic) is str:
                    image_dic = extractRoiFromImg(os.path.join(self.imgDic, img.replace('.png', '.tif')), mask)
                elif type(self.imgDic) is np.ndarray:
                    image_dic = extractRoiFromImg(self.imgDic, mask)
                else:
                    raise TypeError(
                        f"need image path or image array, but got type {type(self.imgDic)} in {self.imgDic}")
                data.image_mcy = self.split(image_mcy)[0]
                data.image_dic = self.split(image_dic)[0]
                if max(data.image_mcy.shape) < cellfilter:
                    print(data.image_mcy.shape)
                    continue
                data.image_mcy = skimage.transform.resize(data.image_mcy, (config.image_width, config.image_height))
                data.image_dic = skimage.transform.resize(data.image_dic, (config.image_width, config.image_height))
                data.image_id = instance
                # yield data
                image_data.append(data)
        return image_data

    def predict_phase(self, images: np.ndarray):
        return self.predictor.predict(images)

    def predict(self, frame):
        image_data = []
        id_data = []
        for cell in self.getCell():
            data = np.dstack([cell.image_dic, cell.image_mcy])
            image_data.append(data)
            id_data.append(cell.image_id)
        images = np.array(image_data)
        print('predict cell count ', len(image_data))
        phases = self.predict_phase(images)
        pl = list(phases)
        print(f"G1/G2: {pl.count('G1/G2')}\nS: {pl.count('S')}\nM: {pl.count('M')}")
        ret = self.parser.addPhase(frame, self.parser.idMap, {frame: dict(zip(id_data, phases))})
        return ret

    def exportResult(self):
        pass


class Segmenter(object):
    def __init__(self, segment_model=None):
        if segment_model is None:
            if config.TIMES == 20:
                self.model = StarDist2D(None, name='segment_20x_model',
                       basedir='saved_segment_models/saved_20x_segment_model')
            elif config.TIMES == 60:
                self.model = StarDist2D(None, name='segment_60x_model',
                       basedir='saved_segment_models/saved_60x_segment_model')
        else:
            self.model = segment_model

    @retry.retry(exceptions=ResourceExhaustedError)
    def segment(self, image: np.ndarray):
        axis_norm = (0, 1)  # normalize channels independently
        im = normalize(image, 1, 99.8, axis=axis_norm)
        labels, details = self.model.predict_instances(im)  # TODO 优化此处限速流程 Optimize the speed limit process here
        return labels, details


class Segmentation(object):
    def __init__(self,
                 image_mcy: np.ndarray,
                 image_dic: np.ndarray,
                 imagename: str,
                 segmenter=None,
                 predictor=None,
                 roi_save_path=None,
                 label_save_path=None):
        self.imageName = imagename
        if segmenter is None:
            self.segmentor = Segmenter()
        else:
            self.segmentor = segmenter
        self.predictor = predictor
        self.roi_save_path = roi_save_path
        self.label_save_path = label_save_path
        self.img_mcy = image_mcy
        self.img_dic = image_dic

    def segment(self):
        return self.segmentor.segment(image=self.img_mcy)

    @property
    def labels(self):
        return self.segment()[0]

    @property
    def details(self):
        return self.segment()[1]

    def rois(self):
        return self.details['coord']

    def save_labels(self, label_save_path=None):
        from csbdeep.io import save_tiff_imagej_compatible
        if label_save_path:
            save_tiff_imagej_compatible(label_save_path, self.labels, axes='YX')
        else:
            save_tiff_imagej_compatible(self.label_save_path, self.labels, axes='YX')

    def save_rois(self, roi_save_path=None):
        from stardist import export_imagej_rois
        if roi_save_path:
            export_imagej_rois(roi_save_path, self.rois())
        elif self.roi_save_path:
            export_imagej_rois(self.roi_save_path, self.rois())
        else:
            raise ValueError('roi_save_path need to be give, in this function or in the class init')

    def __get_segment_result(self):
        """将分割的坐标转化为可识读的json字典"""
        rois = self.rois()
        c = ConverterXY(self.imageName.replace('.tif', '.png'), coordinates=rois)
        return c.json

    def __add_predict_phase(self):
        js = self.__get_segment_result()
        tmp = {}
        predictor = Prediction(mcy=self.img_mcy, dic=self.img_dic, annotation_json=js, imagesize=self.img_mcy.shape,
                               predictor=self.predictor)
        for i in js:
            frame = i
            regions = predictor.predict(frame)  # 限速步骤 # TODO 优化此处限速流程 Optimize the speed limit process here
            tmp[frame] = regions
        return tmp

    @property
    def predict_result(self):
        return self.__add_predict_phase()

    def export_predict_result(self, filepath):
        """导出分割和预测细胞周期到json文件中"""
        with open(filepath, 'w') as f:
            json.dump(self.__add_predict_phase(), f)


def segment(pcna: os.PathLike | str, bf: os.PathLike | str, output: os.PathLike | str, segment_model=None, normalize=False):
    jsons = {}
    global MODEL_USING_INDEX
    mcy_data = readTif(filepath=pcna, normalize=normalize)
    dic_data = readTif(filepath=bf, normalize=normalize)
    segmenter = Segmenter(segment_model=segment_model)
    predictor = Predictor()
    while True:
        try:
            mcy_img, imagename = next(mcy_data)
            dic_img, _ = next(dic_data)
            print(f'start segment {os.path.basename(imagename)} ...')
            start_time = time.time()

            seg = Segmentation(image_mcy=mcy_img,
                               imagename=imagename,
                               image_dic=dic_img,
                               segmenter=segmenter,
                               predictor=predictor)

            value = seg.predict_result
            jsons.update(value)
            end_time = time.time()
            print(f'finish segment {os.path.basename(imagename)}', 'ok')
            print(f'cost time {end_time - start_time}s')
            del seg
            MODEL_USING_INDEX += 1
        except StopIteration:
            break
    json_filename = os.path.basename(pcna).replace('.tif', '.json')
    if output:
        out = output
    else:
        out = json_filename
    with open(out, 'w') as f:
        json.dump(jsons, f)


def convert(img):
    img_mcy = exposure.adjust_gamma(cv2.imread(img, -1), 0.1)
    png = img_as_ubyte(img_mcy)
    return png


if __name__ == '__main__':
    # model = StarDist2D(None, name='stardist_no_shape_completion_60x_seg',
    #                    basedir='/home/zje/CellClassify/ResNet-Tensorflow/models_60x_seg')

    # segment(pcna='/home/zje/CellClassify/predict_data/dataset/beini-dataset/cep192-mcy.tif',
    #         bf='/home/zje/CellClassify/predict_data/dataset/beini-dataset/cep192-dic.tif',
    #         output='/home/zje/CellClassify/predict_data/dataset/beini-dataset/cep192.json', segment_model=None)

    segment(pcna='/home/zje/CellClassify/predict_data/dataset/mcy/copy_of_1_xy05.tif',
            bf='/home/zje/CellClassify/predict_data/dataset/dic/copy_of_1_xy05.tif',
            output='/home/zje/CellClassify/predict_data/dataset/copy_of_1_xy05.json', segment_model=None)

    # segment(pcna='/home/zje/CellClassify/train_dataset/mitosis/series11/mcy/copy11.tif',
    #         bf='/home/zje/CellClassify/train_dataset/mitosis/series11/dic/copy11.tif',
    #         output='/home/zje/CellClassify/train_dataset/mitosis/series1/copy11.json', segment_model=None)

    # segment(pcna='/home/zje/CellClassify/predict_data/dataset/test_pcna.tif',
    #         bf='/home/zje/CellClassify/predict_data/dataset/test_dic.tif',
    #         output='/home/zje/CellClassify/predict_data/dataset/test_result.json', segment_model=model)

    # mcy = '/home/zje/CellClassify/predict_data/dataset/60x_test_predict_phase/mcy'
    # dic = '/home/zje/CellClassify/predict_data/dataset/60x_test_predict_phase/dic'
    # ann = '/home/zje/CellClassify/predict_data/dataset/60x_test_predict_phase/20200729-RPE-s2_cpd.json'
    #
    # p = Predict(mcy=mcy, dic=dic, annotation_json=ann, imagesize=(1200, 1200))
    # p.predict(frame=None)
    pass
