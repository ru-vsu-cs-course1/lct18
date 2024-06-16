import logging.config
from pathlib import Path

from osgeo import gdal
import numpy as np
import cv2 as cv

from lct18 import utils
from lct18.config import config
from lct18.detection import ImageObj, crop_position_detect_rough, normxcorr2

gdal.DontUseExceptions()
logging.config.dictConfig(config.logging.debug)
logger = logging.getLogger()


def load_layout(filename, detect_key_points: bool = False):
    logger.debug('-' * 10)
    logger.debug(f'Loading layout {filename} ...')
    with utils.time_perf_count('layout loading'):
        layout = ImageObj.load_from_geotiff(filename)
        logger.debug(f'Original size: {layout.size}')
    with utils.time_perf_count('layout preprocessing'):
        # layout.normalize()
        layout.resize(config.detection.layout_width_resize_k, config.detection.layout_height_resize_k)
        logger.debug(f'Size for detection: {layout.size}')
        layout.normalize()
        if detect_key_points:
            layout.detect_key_points()
            logger.debug(f'Key points count: {len(layout.kp_points)}')
    return layout


def load_layout_from_png(filename, detect_key_points: bool = False):
    dir = Path(filename).absolute().parent
    filename_without_ext = dir / Path(filename).stem
    logger.debug('-' * 10)
    logger.debug(f'Loading layout {filename_without_ext}.[png, nir.png] ...')
    with utils.time_perf_count('layout loading'):
        layout = ImageObj.load(str(filename_without_ext) + '.png')
        layout.nir = utils.image_load(str(filename_without_ext) + '.nir.png')[1]
    with utils.time_perf_count('layout preprocessing'):
        logger.debug(f'Size for detection: {layout.size}')
        layout.normalize()
        if detect_key_points:
            layout.detect_key_points()
            logger.debug(f'Key points count: {len(layout.kp_points)}')
    layout.original = layout
    return layout


def load_crop(finename, detect_key_points: bool = False):
    logger.debug('-' * 10)
    logger.debug(f'Processing crop {finename} ...')
    with utils.time_perf_count('crop loading and preprocessing'):
        crop = ImageObj.load_from_geotiff(finename)
        logger.debug(f'Original size: {crop.size}')
        # crop.resize(config.detection.crop_width_resize_k, config.detection.crop_height_resize_k)
        # logger.debug(f'Size for detection: {crop.size}')
        # crop.normalize()
        if detect_key_points:
            crop.detect_key_points()
            logger.debug(f'Key points count: {len(layout.kp_points)}')
    return crop


layout_dir = Path(config.data.layout_dir).absolute()
crop_dir = Path(config.data.crop_dir).absolute()

layout_filename = sorted(layout_dir.glob('*.tif'))[0]
layout = ImageObj.load_from_geotiff(layout_filename)
# with utils.time_perf_count('layout.original.normalize()'):
#     layout.original.normalize()
layout.resize(config.detection.layout_width_resize_k, config.detection.layout_height_resize_k)
with utils.time_perf_count('layout.normalize()'):
     layout.normalize()


'''
Работаем в допущениях, что остальные 20 кропов вырезаны абсолютно также.
На самом деле их, конечно, можно найти, но сейчас уже нет времени.

Посчитано при сопоставлении с оригинальной подложкой методом ключевых точек (test2_crops_on_original_layout.py):
Средние значение ширины и высоты кропа, вырезенного из подложки, уменьшенной в 5 раз (2196 x 2196)
232.89322438136023 376.88721259991604
В оригинальной подложке
1164.466121907 1884.436063

Угол поворота рассчитан по точкам: 
  48.54024423, 8.95090983
  2057.41615927, 62.57300834
составляет:
  0.019845838196123
  7.1435640008231 в градусах

Средние разницы составляют:
  6.75832530000002 по x
  3.929742319999997 по y

  33.7916265000001 в исходной подложке
  19.648711599999984 в исходной подложке
  
К общем, будем считать, что в исходнйо подложке разница по x должна быть 1164 x 1884 со смещение 34 и 20
'''


w, h = 232, 377
crop_filename = sorted(crop_dir.glob('*.tif'))[8]
crop = ImageObj.load_from_geotiff(crop_filename)
# некогда сделать по уму
crop.resize(w, h)
crop.normalize()
# layout.resize(config.detection.layout_width_resize_k, config.detection.layout_height_resize_k)


def img_preproc(img: np.array) -> np.array:
    img = utils.histogram_norm(img)
    # хз, почему такие параметры, по идее надо поиграться
    img = cv.morphologyEx(img, cv.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    return img


layout_img = img_preproc(layout.get_channel('gray'))

result_image = cv.cvtColor(layout.get_channel('gray'), cv.COLOR_GRAY2BGR)

for crop_filename in sorted(crop_dir.glob('*0000.tif')):
    crop = ImageObj.load_from_geotiff(crop_filename)
    # некогда сделать по уму
    crop.resize(w, h)
    crop.normalize()

    crop_img = img_preproc(crop.get_channel('gray'))

    # utils.images_show([crop_img, layout_img, layout.get_channel('gray')])

    with utils.time_perf_count('normxcorr2()'):
        corr = normxcorr2(crop_img, layout_img)

    # utils.images_show(corr)
    hot_pos = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
    print(hot_pos)

    x, y = hot_pos[1] - w, hot_pos[0] - h
    rect = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
    result_image = cv.polylines(result_image, [np.int32(rect)], True, (127, 127, 255), 2, cv.LINE_AA)

    # utils.images_show(result_image)

utils.images_show(result_image)
