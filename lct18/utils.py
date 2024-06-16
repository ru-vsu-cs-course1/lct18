import logging
from logging import Logger
from typing import ContextManager, Iterable
from contextlib import contextmanager
from os import PathLike
from time import perf_counter
from typing import Any

from osgeo import gdal, gdal_array
import numpy as np
from numpy import ndarray, dtype
import cv2 as cv
import matplotlib.pyplot as plt


def images_show(images: Iterable[ndarray] | ndarray, titles: Iterable[Any] | Any = None, order: Iterable[int] = None) -> None:
    if isinstance(images, np.ndarray):
        images, titles, order = (images, ), (titles, ), (1, 1)
    titles = titles or []
    titles = [*titles] + [None] * (len(images) - len(titles))
    order = order or (1, len(images))
    fig, ax = plt.subplots(*order, squeeze=False)
    for i, img in enumerate(images):
        title = titles[i] if i < len(titles) else None
        r, c = i // order[1], i % order[1]
        ax[r, c].imshow(img if img.ndim == 2 else cv.cvtColor(img, cv.COLOR_BGR2RGB),
                        cmap=('gray' if img.ndim == 2 else None))
        ax[r, c].title.set_text(str(title or ''))
    plt.tight_layout()
    plt.show()


def image_to_gray(img: ndarray) -> ndarray:
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def image_load_from_geotiff(filename: str | PathLike) -> (Any, ndarray, ndarray, ndarray):
    ds = gdal.Open(str(filename))
    img_data = gdal_array.DatasetReadAsArray(ds)
    bgr, nir, gray = None, None, None
    if len(img_data) >= 3:
        bgr = np.flip(img_data[:3], axis=0)
        bgr = np.moveaxis(bgr, 0, -1)
        # gray = image_to_gray(bgr)
    if len(img_data) > 3:
        nir = img_data[3]
    if len(img_data) == 1 or img_data.ndim == 2:
        gray = img_data
    return ds, bgr, nir, gray


def image_load(filename: str | PathLike) -> (ndarray, ndarray):
    img = cv.imread(str(filename))
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        img = b
    bgr, gray = None, None
    if img.ndim == 3:
        bgr = img
    else:
        gray = img
    return bgr, gray


def image_save(filename: str | PathLike, img: ndarray) -> None:
    cv.imwrite(str(filename), img)


def image_size(img: ndarray) -> (int, int):
    return img.shape[1], img.shape[0]


def image_resize(img: ndarray, width_or_k: float, height_or_k: float) -> ndarray | None:
    if img is None:
        return None

    w, h = image_size(img)
    w_new, h_new = w, h
    if width_or_k and width_or_k != w:
        w_new = round(width_or_k if width_or_k >= 10 else w * width_or_k)
    if height_or_k and height_or_k != h:
        h_new = round(height_or_k if height_or_k >= 10 else h * height_or_k)
    if (w_new, h_new) != (w, h):
        img = cv.resize(img, (w_new, h_new))
    return img


def histogram_norm(img: np.array) -> np.array:
    clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(10, 10))
    if img.ndim == 2:
        img = clahe.apply(img)
    else:
        # b, g, r = cv.split(img)
        # b = clahe.apply(b)
        # g = clahe.apply(b)
        # r = clahe.apply(b)
        # img = cv.merge((b, g, r))
        for i in range(3):
            img[:, :, i] = clahe.apply(img[:, :, i])

    return img


def arr_normalize_min_max(arr: ndarray, min_v: float, max_v: float) -> ndarray:
    return (np.maximum(np.minimum(arr, max_v), min_v) - min_v) / (max_v - min_v)


def image_normalize(img: ndarray | None,
                    min_percentile: int = None, max_percentile: int = None) -> ndarray[Any, dtype[np.uint8]] | None:
    if img is None or img.dtype == np.uint8:
        return img

    min_norm = np.percentile(img, 2 if min_percentile is None else min_percentile, axis=(0, 1))
    max_norm = np.percentile(img, 98 if max_percentile is None else max_percentile, axis=(0, 1))
    # делаем по линиям, весь массив сразу требует много памяти и падает
    for i in range(len(img)):
        img[i] = np.round(arr_normalize_min_max(img[i], min_norm, max_norm) * 255).astype(np.uint8)
    return img.astype(np.uint8)
    # return histogram_norm(img.astype(np.uint8))


def four_points_for_transformation_order(points):
    # возвращает координаты в порядке: верхний левый, верхний правый, нижний правый, нижний левый
    by_y = sorted(points.tolist(), key=lambda x: x[1])
    return np.float32([*sorted(by_y[:2], key=lambda x: x[1]), *sorted(by_y[2:], key=lambda x: -x[1])])


def image_perspective_transform(image, points, new_width: int, new_height: int) -> np.ndarray:
    # как экспериментально выяснилось, если выбрать нужный прямоугольник из картинки
    # и уже его потом трансформировать, то работает на порядки быстрее, чем сразу из исходного изображения

    # допустимы только float32 типы (странно, почему double нельзя)
    src_pos = np.array(points).astype(np.float32)
    min_int_xy = np.floor(np.min(src_pos, axis=0)).astype(np.int32)
    max_int_xy = np.ceil(np.max(src_pos, axis=0)).astype(np.int32)
    image_rect = image[min_int_xy[1]:(max_int_xy[1] + 1), min_int_xy[0]:(max_int_xy[0] + 1)]
    dst_pos = np.float32([
        [0, 0],
        [new_width - 1, 0],
        [new_width - 1, new_height - 1],
        [0, new_height - 1]
    ])
    matrix = cv.getPerspectiveTransform((src_pos - min_int_xy).astype(np.float32), dst_pos)
    return cv.warpPerspective(image_rect, matrix, (new_width, new_height))


@contextmanager
def time_perf_count(code_block_name: Any, logger: Logger = None) -> ContextManager[None]:
    t_start = perf_counter()
    try:
        yield
    finally:
        t_stop = perf_counter()
        logger = logger or logging.getLogger()
        logger.debug(f'Time of [{code_block_name}]: {t_stop - t_start:.06f}')
