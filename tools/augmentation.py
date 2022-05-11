from __future__ import annotations

import os
import skimage
import tifffile
from skimage import transform, exposure, io

import numpy as np
from tqdm import tqdm


class AugmentorV2(object):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.dtype = image.dtype
        self.image = self.__convert_dtype()

    def __convert_dtype(self):
        return skimage.transform.resize(self.__image, self.__image.shape, preserve_range=True).astype(self.dtype)

    def rotate(self, angle):
        return skimage.transform.rotate(self.image, angle=angle, resize=True, preserve_range=True).astype(self.dtype)

    def flipHorizontal(self):
        return np.fliplr(self.image).astype(self.dtype)

    def flipVertical(self):
        return np.flipud(self.image).astype(self.dtype)

    def adjustBright(self, gamma):
        """gamma > 1, bright; gamma < 1. dark"""
        return skimage.exposure.adjust_gamma(self.image, gamma=gamma).astype(self.dtype)

    def movePosition(self):
        right = np.zeros(shape=(self.image.shape[0], 20), dtype=self.dtype)
        tmp1 = np.hstack([self.image, right])
        down = np.zeros(shape=(20, tmp1.shape[1]), dtype=self.dtype)
        tmp2 = np.vstack(tmp1, down)
        return tmp2


def augmentV2(raw_folder, saved_folder):
    for f in tqdm(os.listdir(raw_folder)):
        file = skimage.io.imread(os.path.join(raw_folder, f))
        aug = AugmentorV2(file)
        img_rotate1 = aug.rotate(30)
        img_rotate2 = aug.rotate(60)
        img_rotate3 = aug.rotate(90)
        img_hflip = aug.flipHorizontal()
        img_vflip = aug.flipVertical()
        img_bright = aug.adjustBright(0.7)
        img_dark = aug.adjustBright(1.6)
        img_mv = aug.movePosition()

        savename_rotate1 = os.path.join(saved_folder, f.replace('.tif', '-rotate1.tif'))
        savename_rotate2 = os.path.join(saved_folder, f.replace('.tif', '-rotate2.tif'))
        savename_rotate3 = os.path.join(saved_folder, f.replace('.tif', '-rotate3.tif'))
        savename_hflip = os.path.join(saved_folder, f.replace('.tif', '-hflip.tif'))
        savename_vflip = os.path.join(saved_folder, f.replace('.tif', '-vflip.tif'))
        savename_bright = os.path.join(saved_folder, f.replace('.tif', '-bright.tif'))
        savename_dark = os.path.join(saved_folder, f.replace('.tif', '-dark.tif'))
        savename_mv = os.path.join(saved_folder, f.replace('.tif', '-mv.tif'))
        tifffile.imsave(savename_rotate1, img_rotate1)
        tifffile.imsave(savename_rotate2, img_rotate2)
        tifffile.imsave(savename_rotate3, img_rotate3)
        tifffile.imsave(savename_hflip, img_hflip)
        tifffile.imsave(savename_vflip, img_vflip)
        tifffile.imsave(savename_bright, img_bright)
        tifffile.imsave(savename_dark, img_dark)
        tifffile.imsave(savename_mv, img_mv)


if __name__ == '__main__':
    # b = aug.blur(radius=1.0)
    # plt.imshow(b, cmap='gray')
    # plt.show()
    # augment(r'C:\Users\Frozenleaves\PycharmProjects\resource\20x_train_data\mcy\M')
    src = r'F:\60x_train_data\mcy\M'
    dst = r'F:\60x_train_data\augment_mcy\M'
    if not os.path.exists(dst):
        os.makedirs(dst)
        augmentV2(src, dst)
