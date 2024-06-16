import argparse
import sys
from pathlib import Path

from osgeo import gdal
import numpy as np

from lct18 import utils
from lct18.config import config
from lct18.detection import ImageObj

gdal.DontUseExceptions()

conf_d = config.detection


def parse_args():
    parser = argparse.ArgumentParser(prog='to_png', description='Convert tiff to png', conflict_handler='resolve')
    parser.add_argument('in_file', help='input tiff file')
    parser.add_argument('out_file', help='output png file', nargs='?')
    parser.add_argument('-c', '--channel', help='image channel | rgb | all',
                        choices=['r', 'g', 'b', 'nir', 'rgb', 'gray', 'all'], default='rgb')
    parser.add_argument('-l', '--log', help='print log', action='store_true')
    parser.add_argument('--show', help='show result image', action='store_true')
    parser.add_argument('-w', '--width', help='out width or resize coefficient', type=float)
    parser.add_argument('-h', '--height', help='out height or resize coefficient', type=float)
    cmd_args = parser.parse_args()
    return cmd_args


cmd_args = parse_args()
if cmd_args.channel == 'all' and cmd_args.out_file:
    print('The extraction of all channels is not compatible with the one output png filename', file=sys.stderr)
    exit(1)

def log(*args, **kwds):  # noqa
    global cmd_args
    if cmd_args.log:
        print(*args, **kwds)


in_file = Path(cmd_args.in_file).absolute()
log(f'Input file: {in_file}')
image = ImageObj.load_from_geotiff(in_file)
image.parent = image

width_src, height_src = (image.parent or image).width, (image.parent or image).height
log(f'Src size: width = {image.width}, height = {image.height}')

width_dst, height_dst = cmd_args.width or width_src, cmd_args.height or height_src
if (width_dst, height_dst) != (width_src, height_src):
    image.resize(width_dst, height_dst)
    log(f'Resized: width = {image.width}, height = {image.height}')

if cmd_args.log:
    min_v, max_v, min_norm, max_norm = [], [], [], []
    if image.bgr is not None:
        min_v.extend(np.flip(np.min(image.bgr, axis=(0, 1))).tolist())
        max_v.extend(np.flip(np.max(image.bgr, axis=(0, 1))).tolist())
        min_norm.extend(np.flip(np.percentile(image.bgr, conf_d.normalize_min_percentile, axis=(0, 1))).astype(int))
        max_norm.extend(np.flip(np.percentile(image.bgr, conf_d.normalize_max_percentile, axis=(0, 1))).astype(int))
    elif image.gray is not None:
        min_v.append(np.min(image.gray))
        max_v.append(np.max(image.gray))
        min_norm.append(int(np.percentile(image.gray, conf_d.normalize_min_percentile)))
        max_norm.append(int(np.percentile(image.gray, conf_d.normalize_max_percentile)))
    if image.nir is not None:
        min_v.append(np.min(image.nir))
        max_v.append(np.max(image.nir))
        min_norm.append(int(np.percentile(image.nir, conf_d.normalize_min_percentile)))
        max_norm.append(int(np.percentile(image.nir, conf_d.normalize_max_percentile)))

    log(f'Min-max channels values:\n  {', '.join(str(v) for v in min_v)}\n  {', '.join(str(v) for v in max_v)}')
    log(f'Min-max channels norm values:\n  {', '.join(str(v) for v in min_norm)}\n  {', '.join(str(v) for v in max_norm)}')

image.normalize()

channels = ['rgb'] if not cmd_args.channel else \
    ['rgb', 'r', 'g', 'b', 'gray', 'nir'] if cmd_args.channel == 'all' else [cmd_args.channel]
for ch in channels:
    ch_img = utils.image_normalize(image.bgr) if ch == 'rgb' else image.get_channel(ch, normalize=True)
    if ch_img is not None:
        out_file = Path(cmd_args.out_file or in_file.with_suffix(('' if ch == 'rgb' else f'.{ch}') + '.png')).absolute()
        utils.image_save(out_file, ch_img)
        log(f'Output file: {out_file}')
        if cmd_args.show:
            utils.images_show(ch_img, out_file)
