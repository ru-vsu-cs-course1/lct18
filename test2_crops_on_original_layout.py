import logging.config
from pathlib import Path

from osgeo import gdal
import numpy as np
import cv2 as cv

from lct18 import utils
from lct18.config import config
from lct18.detection import ImageObj, crop_position_detect_rough

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
        crop.resize(config.detection.crop_width_resize_k, config.detection.crop_height_resize_k)
        logger.debug(f'Size for detection: {crop.size}')
        crop.normalize()
        if detect_key_points:
            crop.detect_key_points()
            logger.debug(f'Key points count: {len(layout.kp_points)}')
    return crop


layout_dir = Path(config.data.layout_dir).absolute()
crop_dir = Path(config.data.crop_dir).absolute()

filename = layout_dir / config.data.layout_main_file
# filename = layout_dir / 'layout_2021-10-10.tif'
layout = load_layout(filename, True)
# layout = load_layout_from_png(filename, True)

result_img = layout.bgr

ww, hh = 0, 0
pp = []

for f in sorted(crop_dir.glob('*0000.tif')):
    crop = load_crop(f, True)

    pos, good_matches, crop_kp_points, layout_kp_points = crop_position_detect_rough(crop, layout)

    w = np.linalg.norm(np.array(pos[0]) - np.array(pos[1]))
    h = np.linalg.norm(np.array(pos[0]) - np.array(pos[3]))
    ww += w
    hh += h
    pp.append(pos)

    logger.debug(f'Crop position: {', '.join(f'({x[0]:.06f}, {x[1]:.06f})' for x in pos)}')
    logger.debug(f'Diffs: {(pos[1][0] - pos[0][0]):.06f}, {(pos[2][0] - pos[3][0]):.06f}, {(pos[3][1]-pos[0][1]):.06f}, {(pos[2][1] - pos[1][1]):.06f}')
    result_img = cv.polylines(result_img, [np.int32(pos)], True, (255, 255, 255), 2, cv.LINE_AA)

print(pp)
print(ww / len(pp), hh / len(pp))

utils.image_save(crop_dir.parent / 'result' / 'crops_on_layout.png', result_img)
utils.images_show(result_img)
