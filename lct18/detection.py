import logging
from os import PathLike
from typing import Any, Self

import numpy as np
from numpy import ndarray
import cv2 as cv
import rtree
from scipy.signal import fftconvolve

from lct18 import utils
from lct18.config import config
from lct18.utils import time_perf_count


logger = logging.getLogger('detection')

key_points_detector = cv.BRISK_create()  # noqa
# используем Хемминга, т.к. в BRISK дескрипторы бинарные
key_points_matcher = cv.BFMatcher(cv.NORM_HAMMING)


class ImageObj:

    def __init__(self, ds: Any | None, bgr: ndarray | None, nir: ndarray | None, gray: ndarray | None):
        self.original: Self | None = None
        self.ds, self.bgr, self.nir, self.gray = ds, bgr, nir, gray
        arr: ndarray | None = None
        self.r = self.g = self.b = arr
        self.kp_detector = key_points_detector
        self.kp_points: list | None = None
        self.kp_descriptors: list | None = None

    @classmethod
    def load(cls, filename: str | PathLike) -> Self:
        bgr, gray = utils.image_load(filename)
        return cls(None, bgr, None, gray)

    @classmethod
    def load_from_geotiff(cls, filename: str | PathLike) -> Self:
        return cls(*utils.image_load_from_geotiff(filename))

    @property
    def size(self) -> list[int]:
        img = self.bgr if self.bgr is not None else self.gray
        return utils.image_size(img) if img is not None else (0, 0)

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def resize(self, width_or_k: float = None, height_or_k: float = None):
        width_or_k = width_or_k or 1
        height_or_k = height_or_k or 1
        if width_or_k != 1 or height_or_k != 1:
            self.original = ImageObj(self.ds, self.bgr, self.nir, self.gray)
            self.ds = None
            self.bgr = utils.image_resize(self.original.bgr, width_or_k, height_or_k)
            self.nir = utils.image_resize(self.original.nir, width_or_k, height_or_k)
            self.gray = utils.image_resize(self.original.gray, width_or_k, height_or_k)
            self.r = utils.image_resize(self.original.r, width_or_k, height_or_k)
            self.g = utils.image_resize(self.original.g, width_or_k, height_or_k)
            self.b = utils.image_resize(self.original.b, width_or_k, height_or_k)

    def normalize(self) -> None:
        percentiles = config.detection.normalize_min_percentile, config.detection.normalize_max_percentile
        self.bgr = utils.image_normalize(self.bgr, *percentiles)
        self.nir = utils.image_normalize(self.nir, *percentiles)
        self.gray = utils.image_normalize(self.gray, *percentiles)
        self.r = utils.image_normalize(self.r, *percentiles)
        self.g = utils.image_normalize(self.r, *percentiles)
        self.b = utils.image_normalize(self.r, *percentiles)

    def get_channel(self, channel: str = None, normalize: bool = False) -> ndarray | None:
        channel = str(channel).lower()
        if channel in ('r', 'red', 'g', 'green', 'b', 'blue') and self.bgr is None:
            return None
        if channel in ('gray', 'grey') and self.bgr is None and self.gray is None:
            return None

        if channel in ('nir', 'n', '4'):
            if self.nir is None:
                return None
            img = self.nir
        elif channel in ('r', 'red'):
            img = self.r = self.r if self.r is not None else self.bgr[:, :, 2]
        elif channel in ('g', 'green'):
            img = self.g = self.g if self.g is not None else self.bgr[:, :, 1]
        elif channel in ('b', 'blue'):
            img = self.b = self.b if self.b is not None else self.bgr[:, :, 0]
        elif channel in ('gray', 'grey'):
            img = self.gray = self.gray if self.gray is not None else utils.image_to_gray(self.bgr)
        else:
            return None

        if normalize:
            img = utils.image_normalize(img)
        return img

    def detect_key_points(self, channel: str = None) -> None:
        if self.kp_points:
            return
        img = self.bgr if str(channel).lower() in ('rgb', 'bgr', 'all', 'color') else \
            self.get_channel(channel or 'gray')
        self.kp_points, self.kp_descriptors = \
            self.kp_detector.detectAndCompute(utils.image_normalize(img), None)

    def save(self, filename: str | PathLike, channel=None, normalize: bool = True) -> ndarray | None:
        if not channel:
            img = self.bgr if self.bgr is not None else self.gray
        else:
            channel = str(channel).lower()
            img = self.get_channel(channel, normalize)

        if img is not None:
            utils.image_save(filename, img)
        return


def crop_position_detect_rough(crop: ImageObj, layout: ImageObj, channel: str = None) -> Any:
    conf_d = config.detection

    if not layout.kp_points:
        with time_perf_count('crop key points detection', logger):
            layout.detect_key_points(channel=channel)
    if not crop.kp_points:
        with time_perf_count('crop key points detection', logger):
            crop.detect_key_points(channel=channel)

    key_points_matcher = cv.BFMatcher(cv.NORM_HAMMING)
    matcher = key_points_matcher
    with time_perf_count('key points matching', logger):
        # matches = matcher.knnMatch(crop.kp_descriptors, layout.kp_descriptors[:len(layout.kp_descriptors)//2], k=conf_d.knn_match_n)
        matches = matcher.knnMatch(crop.kp_descriptors, layout.kp_descriptors, k=conf_d.knn_match_n)
        matches = sorted(matches, key=lambda gr: min(m.distance for m in gr))
        logger.debug(f'Matches count: {sum(len(m) for m in matches)}')
        # matches = np.array(matches[:300]).flatten().tolist()
        matches = np.array(matches[:conf_d.knn_match_select_groups_n]).flatten().tolist()
        logger.debug(f'Selected matches count: {len(matches)}')
        layout_to_crop_kp_idxs = {m.trainIdx: m.queryIdx for m in matches}

    with time_perf_count('key points rects iteration with rtree', logger):
        rtree_idx = rtree.index.Index()
        for matrix in matches:
            rtree_idx.insert(matrix.trainIdx, layout.kp_points[matrix.trainIdx].pt)

        dx, dy = crop.width * conf_d.search_rect_width_step_k, crop.height * conf_d.search_rect_height_step_k
        w, h = crop.width * conf_d.search_rect_width_resize_k, crop.height * conf_d.search_rect_height_resize_k
        rects = []
        for x in np.arange(0, layout.width, dx):
            for y in np.arange(0, layout.height, dy):
                search_rect = (x, y, x + w, y + h)
                points = set(p for p in rtree_idx.intersection(search_rect))
                # uniq_crop_kp_count = len(points)
                uniq_crop_kp_count = len(set(layout_to_crop_kp_idxs[p] for p in points))
                rects.append((x, y, uniq_crop_kp_count, points))

    with time_perf_count('key points rects filtering', logger):
        rects.sort(key=lambda r: r[2], reverse=True)
        rects_filtered = [rects[0]]
        for r1 in rects:
            same = False
            for r2 in rects_filtered:
                if len(r2[3] & r1[3]) > conf_d.cross_points_same_area_k * len(r2[3]):
                    same = True
                    break
            if not same:
                rects_filtered.append(r1)

    # здесь берем первый прямоугольник по максимальному количеству точек,
    # по-хорошему, надо рассмотреть несколько первых и уже проверять соответствие другими методами
    logger.debug(f'Key points count in first rects: {", ".join(str(r[2]) for r in rects_filtered[:5])}, ...')
    best_rect = rects_filtered[0]
    good_matches = [m for m in matches if m.trainIdx in best_rect[3]]

    crop_kp_points = np.float32([crop.kp_points[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    layout_kp_points = np.float32([layout.kp_points[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    with time_perf_count('findHomography for key points matches', logger):
        matrix, mask = cv.findHomography(crop_kp_points, layout_kp_points, cv.RANSAC, 5.0)

    pos_c = np.double([
        [0, 0],
        [crop.width - 1, 0],
        [crop.width - 1, crop.height - 1],
        [0, crop.height - 1]
    ]).reshape(-1, 1, 2)
    pos_on_layout = cv.perspectiveTransform(pos_c, matrix)[:, 0, :]

    return pos_on_layout, good_matches, crop.kp_points, layout.kp_points


# взято из https://github.com/Sabrewarrior/normxcorr2-python
def normxcorr2(template, image, mode="full"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        logger.warning("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out
