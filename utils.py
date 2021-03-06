"""
适用于图像分割处理的各种函数

"""

from __future__ import annotations
from typing import ClassVar, Tuple, List
import os
import typing
import hashlib
from copy import deepcopy
import skimage.exposure as exposure
import skimage.transform
from skimage.util import img_as_ubyte
import json
from libtiff import TIFF
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import config

ArrayLike = typing.Union[np.ndarray, list]

PHASE_MAP = {
    "G1/G2": 0,
    "M": 1,
    "S": 2
}


def tif2png(img: str, png_dir, gamma=0.1):
    """将tif stack 转化为png"""
    tif = TIFF.open(img)
    index = 0
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    for i in tif.iter_images():
        filename = os.path.basename(img).replace('.tif', '-' + str(index).zfill(4) + '.png')
        img_mcy = exposure.adjust_gamma(i, gamma)
        png = img_as_ubyte(img_mcy)
        plt.imsave(os.path.join(png_dir, filename), png, cmap='gray')
        index += 1


def find_positions(array):
    """
     寻到到每个细胞的外接矩形边界坐标
     :param array: 单个细胞图像矩阵
     :return: 顶点坐标
    """
    start_0 = None
    index = 0
    for row in enumerate(array):
        if len(np.unique(row[1])) > 1:
            start_0 = row[0]
            break
    for row in array:
        if len(np.unique(row)) > 1:
            index += 1
    start_1 = start_0 + index
    return start_0, start_1


#def find_positions(array, blank=0):
#    """
#    寻到到每个细胞的外接矩形边界坐标
#    :param array: 单个细胞图像矩阵
#    :param blank: 留白边界
#    :return: 顶点坐标
#    """
#    start_0 = None
#    index = 0
#    for row in enumerate(array):
#        if len(np.unique(row[1])) > 1:
#            start_0 = row[0]
#            break
#    if start_0 - blank <= 0:
#        start_0 = 0
#    else:
#        start_0 -= blank
#    for row in array:
#        if len(np.unique(row)) > 1:
#            index += 1
#    start_1 = start_0 + index
#    if start_1+blank >= array.shape[0]:
#        start_1 = array.shape[1]
#    else:
#        start_1 += blank
#    return start_0, start_1


def split_target(array):
    """
    从背景图像中裁剪出包含目标细胞的最小外接矩形
    """
    x0, x1 = find_positions(array)
    y0, y1 = find_positions(array.T)
    return array[x0: x1, y0: y1], np.max([x1 - x0, y1 - y0])


def divideImage(img: ArrayLike, m=2, n=2) -> List[np.ndarray]:
    result = []
    data = np.array(img)
    hsplit_ret = np.hsplit(data, m)
    for i in hsplit_ret:
        ret = np.vsplit(i, n)
        result.append(ret[0])
        result.append(ret[1])
    return result


def json2mask(filepath, raw, mask):
    """
    :param filepath:  需要转化的json文件路径
    :param raw: json文件包含的图片所在目录
    :param mask: 生成的掩膜文件保存目录
    :return: 如果转化成功，返回True
    for example:
    """
    annotation = json.load(open(filepath, 'r'))
    for i in annotation:
        filename = i
        regions = annotation[i].get('regions')
        image_path = os.path.join(raw, filename)
        image = cv2.imread(image_path, -1)  # image = skimage.io.imread(image_path)
        height, width = image.shape[:2]
        mask_arr = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            polygons = region.get('shape_attributes')
            points = []
            for j in range(len(polygons['all_points_x'])):
                x = int(polygons['all_points_x'][j])
                y = int(polygons['all_points_y'][j])
                points.append((x, y))
            contours = np.array(points)
            cv2.fillConvexPoly(mask_arr, contours, (255, 255, 255))
        save_path = os.path.join(mask, filename)
        cv2.imwrite(save_path, mask_arr)


def coordinate2mask(coords: np.ndarray | list | tuple, image_size: Tuple[int, int] = None) -> \
        typing.List[np.ndarray] | np.ndarray:
    """根据轮廓坐标绘制mask, 如果只传入一组轮廓坐标值，请务必将其置于列表中传入函数，
    例如， coord = ([x1 x2 ... xn], [y1 y2 ... yn]),调用时请按照coordinate2mask([coord])调用
    """
    results = []
    for coord in coords:
        if image_size is None:
            mask = np.zeros(config.imageSize, dtype=np.uint8)
        else:
            mask = np.zeros(image_size, dtype=np.uint8)
        points = []
        for j in range(len(coord[0])):
            x = int(coord[0][j])
            y = int(coord[1][j])
            points.append((x, y))
        contours = np.array(points)
        cv2.fillConvexPoly(mask, contours, (1, 1, 1))
        results.append(mask)
    return results


def extractRoiFromImg(images: str | np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    根据掩膜提取原始图像中的区域, 注意只能是单通道图，如果是rgb，请先转化为灰度图
    :param images: 原始图像
    :param mask: 掩膜文件
    :return: 单个细胞图像数据
    """
    if type(images) is str:
        src = cv2.imread(images, -1)
    else:
        src = images
    dst = np.zeros_like(src)
    cv2.copyTo(src, mask, dst)
    return dst


class JsonParser(object):
    """解析标注json文件"""

    def __init__(self, file: str | dict | np.ndarray):
        if type(file) is str:
            assert file.endswith('.json')
            with open(file) as fp:
                self.json = json.load(fp)
        else:
            self.json = file
        self.idMap = self.setIdForCell()
        self.phaseMap = self.setPhaseMap()
        self.images = list(self.json.keys())

    def __len__(self):
        return len(self.json)

    @staticmethod
    def addPhase(frame, id_map, phase_map):
        """添加预测周期"""
        region_tmp = {"shape_attributes": {
            "name": "polygon", 'all_points_x': None, 'all_points_y': None},
            "region_attributes": {
                "phase": None
            }
        }
        regions = []
        for i in id_map[frame]:
            all_x, all_y = id_map[frame][i]
            phase = phase_map[frame].get(i)
            if phase is None:
                continue
            r = deepcopy(region_tmp)
            r["shape_attributes"]['all_points_x'] = all_x
            r["shape_attributes"]['all_points_y'] = all_y
            r["region_attributes"]["phase"] = phase
            regions.append(r)
        return {'filename': frame, "size": 1440000, "regions": regions, "file_attributes": {}}

    @staticmethod
    def getIdFromCoordinate(coordinate):
        return hashlib.md5(str(coordinate).encode('utf-8')).hexdigest()

    def setIdForCell(self):
        """为每个细胞设置一个独一无二的id， 依据的是细胞的坐标, id为坐标数组的MD5值，
        同时，添加全局变量idMap字典，细胞id为键，坐标为值
        """
        ID_MAP = {}
        for key in self.json:
            coords = self.getCoordinates(key)
            id_map = {}
            for co in coords:
                id_map[self.getIdFromCoordinate(co)] = co
            ID_MAP[key] = id_map
        return ID_MAP

    def getCoordinate(self):
        """获取某个细胞的轮廓坐标"""
        pass

    def getCoordinates(self, key):
        """获取单张图片中所有细胞实例的轮廓坐标"""
        regions = self.json.get(key).get('regions')
        coords = [(i['shape_attributes']['all_points_x'], i['shape_attributes']['all_points_y']) for i in regions]
        return coords

    @staticmethod
    def coordinate2contours(coord):
        points = []
        for j in range(len(coord[0])):
            x = int(coord[0][j])
            y = int(coord[1][j])
            points.append((x, y))
        contour = np.array(points)
        return contour

    @staticmethod
    def getContourArea(contour):
        """返回细胞轮廓的面积大小"""
        return cv2.contourArea(contour)

    def __getPhase(self, key):
        """获取某个细胞的细胞周期
        :key 每一帧的文件名
        """
        regions = self.json.get(key).get('regions')
        phase_map = {}
        for region in regions:
            try:
                phase = region['region_attributes']['phase']
            except KeyError:
                phase = 'E'
            coord = (region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y'])
            phase_map[self.getIdFromCoordinate(coord)] = "".join(phase.split())
        return phase_map

    def getPhase(self, image, id_=None, coord=None):
        if id:
            return self.phaseMap[image][id_]
        elif coord:
            return self.phaseMap[image][self.getIdFromCoordinate(coord)]

    def setPhase(self, phase, id_=None, coord=None):
        if id:
            self.phaseMap[id_] = phase
        elif coord:
            self.phaseMap[self.getIdFromCoordinate(coord)] = phase

    def setPhaseMap(self):
        phaseMap = {}
        for key in self.json:
            phaseMap[key] = self.__getPhase(key)
        return phaseMap

    @property
    def imageName(self):
        return self.images


class Data(object):
    image_mcy: ClassVar[np.ndarray] = None
    image_dic: ClassVar[np.ndarray] = None
    image_id: ClassVar[str] = None
    phase: ClassVar[str] = None


class Augment(object):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.image = self.__convert_dtype()

    def __convert_dtype(self):
        return skimage.transform.resize(self.__image, self.__image.shape)

    def rotate(self, angle):
        return skimage.transform.rotate(self.image, angle=angle, resize=True)

    def flipHorizontal(self):
        return np.fliplr(self.image)

    def flipVertical(self):
        return np.flipud(self.image)

    def adjustBright(self, gamma):
        """gamma > 1, bright; gamma < 1. dark"""
        return skimage.exposure.adjust_gamma(self.image, gamma=gamma)

    def addNoise(self):
        return skimage.util.random_noise(self.image, mode='gaussian')


def split(array):
    """返回图像中包含单个细胞的最小外接矩形区域"""
    x0, x1 = find_positions(array)
    y0, y1 = find_positions(array.T)
    return array[x0: x1, y0: y1], np.max([x1 - x0, y1 - y0])


class DataGenerator(object):
    """
    为分类算法生成训练数据，其数据结构为X_train = [img array], Y_train = [phase list]
    """

    def __init__(self, training_data_mcy, training_label, train_data_dic=None, imagesize=None):
        """每一个parser对应一个json数据集，其中包含多张图像
        :param training_data*: 存放tif图像文件所在文件路径
        :param training_label: 对应的json标注文件所在路径
        """
        self.imagesize = imagesize
        self.trainingDataMcy = training_data_mcy
        self.trainingDataDic = train_data_dic  # 增加dic通道训练数据
        self.trainingLabel = training_label
        self.parser = JsonParser(training_label)
        self.images = None
        self.datas = None

    def generate(self, normalization=False):
        """生成Data实例返回
        """
        images = self.parser.images
        for img in tqdm(images, desc="data generate "):
            instances = self.parser.idMap.get(img)
            phases = self.parser.phaseMap.get(img)
            for instance in instances:
                data = Data()
                coord = instances[instance]
                mask = coordinate2mask([coord], image_size=self.imagesize)[0]
                image_mcy = extractRoiFromImg(os.path.join(self.trainingDataMcy, img.replace('.png', '.tif')), mask)
                if self.trainingDataDic:
                    image_dic = extractRoiFromImg(os.path.join(self.trainingDataDic, img.replace('.png', '.tif')), mask)
                    data.image_dic = split(image_dic)[0]
                else:
                    image_dic = None
                    data.image_dic = image_dic
                data.image_mcy = split(image_mcy)[0]
                data.phase = PHASE_MAP.get(phases.get(instance))
                data.image_id = instance
                # 筛选并排除周期设置为E的细胞
                if data.phase is None:
                    continue
                else:
                    if normalization:
                        # 像素值归一化到[0, 1]
                        data.image_mcy = cv2.normalize(data.image_mcy, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
                        if self.trainingDataDic:
                            data.image_dic = cv2.normalize(data.image_dic, None, 0, 1, cv2.NORM_MINMAX).astype(
                                np.float32)
                    yield data


class ConverterXY(object):
    """
    根据掩膜的坐标，将其转化为via可识读的json文件
    for example:
    >>> co = [[1, 2, 3, 4], [5, 6, 7, 8]]
    >>> img = 'test.png'
    >>> con = ConverterXY(img, co)
    >>> with open('test.json', 'w') as f:
    >>>     json.dump(con.json, f)
    """

    def __init__(self, img_name: str, coordinates: ArrayLike, phase: dict = None):
        self.region_template = {
            "shape_attributes":
                {
                    "name": "polygon",
                    "all_points_x": [],
                    "all_points_y": []
                },
            "region_attributes":
                {
                    "phase": "G1/G2"
                }
        }
        self.img_name = img_name
        self.coordinates = coordinates
        self.json_template = {
            self.img_name: {
                "filename": "",
                "size": 1440000,
                "regions": [],
                "file_attributes": {}
            }
        }
        self.phase = phase

    def generate_regions(self, X, Y, phase='G1/G2'):
        region = deepcopy(self.region_template)
        shape_attr = region.get('shape_attributes')
        region_attr = region.get('region_attributes')
        region_attr['phase'] = phase
        shape_attr['all_points_x'] = X
        shape_attr['all_points_y'] = Y
        return region

    def roi2json(self):
        roi_json = deepcopy(self.json_template)
        regions = []
        XY = self.coordinates
        for i in range(len(XY)):
            # for xy in XY:
            all_y, all_x = XY[i]
            if self.phase is not None:
                ph = self.phase[i]  # phase应该是一个字典{1: 'phase1', 2: 'phase'}或者列表
                region = self.generate_regions(all_x.tolist(), all_y.tolist(), phase=str(ph))
            else:
                region = self.generate_regions(all_x.tolist(), all_y.tolist())
            regions.append(region)
        roi_json[self.img_name]['regions'] = regions
        roi_json[self.img_name]['filename'] = self.img_name
        return roi_json

    @property
    def json(self):
        return self.roi2json()


if __name__ == '__main__':
    #     d3 = DataGenerator(r'C:\Users\Frozenleaves\PycharmProjects\resource\dataset\tif\mcy',
    #                        r'C:\Users\Frozenleaves\PycharmProjects\resource\dataset\manual_annotation_1221.json',
    #                        train_data_dic=r'C:\Users\Frozenleaves\PycharmProjects\resource\dataset\tif\dic')
    #     x = d3.generate()
    pass
