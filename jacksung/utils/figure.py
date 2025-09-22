import sys
import math
import cv2
import shutil
from jacksung.utils.multi_task import ThreadingLock
from PIL import ImageFont
from osgeo import gdal, osr
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint
import os
from jacksung.utils.data_convert import nc2np, np2tif
from jacksung.utils.image import crop_png, zoom_image, zoomAndDock, draw_text, concatenate_images, make_block
from jacksung.utils.cache import Cache
import rasterio
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from rasterio.transform import from_origin
import yaml
import argparse
import jacksung.utils.fastnumpy as fnp
from tqdm import tqdm
from datetime import datetime, timedelta
from matplotlib.ticker import MaxNLocator
import netCDF4 as nc
import math
from jacksung.utils.data_convert import np2tif, Coordinate


def _get_color_normalization(data, colors):
    max_value = colors[-1][0]
    min_value = colors[0][0]
    data[data < min_value] = min_value
    data[data > max_value] = max_value
    data = (data - min_value) / (max_value - min_value)
    new_colors = []
    for color in colors:
        new_colors.append([(color[0] - min_value) / (max_value - min_value), color[1]])
    return data, new_colors


# 色带颜色定位
def _get_color_position(value, colors):
    colors_min, colors_max = colors[0][0], colors[-1][0]
    colors = [[(color[0] - colors_min) / (colors_max - colors_min), color[1]] for color in colors]
    i = 0
    while i < len(colors) - 1:
        if value <= colors[i + 1][0]:
            break
        i += 1
    color_str0, color_str1 = colors[i][1], colors[i + 1][1]
    r1, g1, b1, r2, g2, b2 = int(color_str0[1:3], 16), int(color_str0[3:5], 16), int(color_str0[5:7], 16), \
        int(color_str1[1:3], 16), int(color_str1[3:5], 16), int(color_str1[5:7], 16)
    r = (value - colors[i][0]) / (colors[i + 1][0] - colors[i][0]) * (r2 - r1) + r1
    g = (value - colors[i][0]) / (colors[i + 1][0] - colors[i][0]) * (g2 - g1) + g1
    b = (value - colors[i][0]) / (colors[i + 1][0] - colors[i][0]) * (b2 - b1) + b1
    return np.array((b, g, r))


def make_color_map(colors, h, w, unit='', l_margin=300, r_margin=200, font_size=150):
    colors_map = np.zeros((h, w, 3), dtype=np.uint8) + 255
    w = w - l_margin - r_margin
    for i in range(l_margin, w + l_margin):
        i = i - l_margin
        colors_map[:h - 150, i + l_margin] = _get_color_position(i / w, colors)
        if i in [0, w // 2, w - 1]:
            text = str(round((i / w) * (colors[-1][0] - colors[0][0]) + colors[0][0]))
            if i == 0:
                text += unit
            colors_map = draw_text(colors_map, (i - 100 + l_margin, h - 150),
                                   font=ImageFont.truetype(
                                       rf'{os.path.abspath(os.path.dirname(__file__))}/../libs/times.ttf', font_size),
                                   text=text)
    return colors_map


def _make_fig(file_np,
              # [经度起,经度止,经度步长],[纬度起,纬度止,纬度步长]
              # np数据会自动填充整个图形,确保数据范围和area范围一致
              area, file_title='', save_name='img1.png',
              # 色带范围,请给出实际的数据范围
              # color=((0, '#1E90FF'), (2, '#1874CD'), (5, '#3A5FCD'), (10, '#0000CD'), (30, '#9400D3')),
              colors=None,
              # 字体大小
              font_size=15,
              # 放大区域
              zoom_rectangle=(310 * 5, 300 * 5, 50 * 5, 40 * 5),
              # 放大区域停靠位置
              zoom_docker=(300, 730),
              # 图片清晰度
              dpi=500,
              xy_axis=None,
              # 添加各种特征
              # 自然海岸界,其他自带要素的参考cartopy
              # '10m', '50m', or '110m'
              features=(
                      cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='none',
                                                   linewidth=0.4), cfeature.OCEAN, cfeature.LAND, cfeature.RIVERS),
              border_type=None, make_fig_lock=None):
    # corp = [92, 31, 542, 456]
    if xy_axis is None:
        xy_axis = area
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 设置经纬度范围,限定为中国
    # 注意指定crs关键字,否则范围不一定完全准确
    extents = [area[0][0], area[0][1], area[1][0], area[1][1]]
    if area[0][1] > 180:
        central_longitude = 180
    else:
        central_longitude = 0
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    if make_fig_lock is not None:
        make_fig_lock.acquire()
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extents, crs=proj)
    if colors is None:
        np_max, np_min = np.nanmax(file_np), np.nanmin(file_np)
        break_value = (np_max - np_min) / 4
        colors = ((np_min, '#1E90FF'), (np_min + break_value, '#1874CD'), (np_min + 2 * break_value, '#3A5FCD'),
                  (np_min + 3 * break_value, '#0000CD'), (np_max, '#9400D3'))
    # 用色带给数据上色,输入单通道,返回三通道图
    elevation, new_colors = _get_color_normalization(file_np, colors)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', new_colors)
    for feature in features:
        ax.add_feature(feature)
    ax.imshow(elevation, origin='upper', extent=extents, transform=proj, cmap=cmap)
    # 添加网格线
    if border_type is not None:
        # ax.gridlines(line_style='--')
        ax.gridlines(line_style=border_type)
    # 设置大刻度和小刻度
    tick_proj = ccrs.PlateCarree()
    ax.set_xticks(np.arange(xy_axis[0][0], xy_axis[0][1] + 1, xy_axis[0][2]), crs=tick_proj)
    # ax.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=tick_proj)
    ax.set_yticks(np.arange(xy_axis[1][0], xy_axis[1][1] + 1, xy_axis[1][2]), crs=tick_proj)
    # ax.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=tick_proj)
    # 利用Formatter格式化刻度标签
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(file_title, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.title(fontsize=font_size)
    plt.savefig(save_name)
    if zoom_rectangle is not None:
        read_png = cv2.imread(save_name)
        read_png = zoomAndDock(read_png, zoom_rectangle, zoom_docker, scale_factor=5, border=14)
        cv2.imwrite(save_name, read_png)
    np_data = cv2.imread(save_name) - 255
    np_sum_h = np.nonzero(np_data.sum(axis=(1, 2)))[0]
    np_sum_w = np.nonzero(np_data.sum(axis=(0, 2)))[0]
    # print(np_sum_h, np_sum_w)
    crop_png(save_name, left=min(np_sum_w[0], 400), top=min(np_sum_h[0], 150), right=np_sum_w[-1], bottom=np_sum_h[-1])
    plt.close()
    if make_fig_lock is not None:
        make_fig_lock.release()
    return colors
    # plt.show()


def make_fig(data,
             area,
             xy_axis=None,
             file_title='',
             save_name='figure_default.png',
             colors=None,
             font_size=15,
             zoom_rectangle=None,
             zoom_docker=(300, 730),
             dpi=500,
             features=(
                     cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='none',
                                                  linewidth=0.4), cfeature.OCEAN, cfeature.LAND, cfeature.RIVERS),
             border_type=None,
             colormap_l_margin=300,
             colormap_r_margin=200,
             colormap_unit=''):
    colors = _make_fig(data, font_size=font_size, zoom_rectangle=zoom_rectangle, zoom_docker=zoom_docker, dpi=dpi,
                       features=features, border_type=border_type, xy_axis=xy_axis,
                       file_title=file_title, save_name=save_name, area=area, colors=colors)
    img = cv2.imread(save_name)
    h, w, c = img.shape
    cm = make_color_map(colors, 180, w, unit=colormap_unit, l_margin=colormap_l_margin, r_margin=colormap_r_margin)
    white_block = make_block(10, w)
    merge_img = concatenate_images([img, white_block, cm], direction='v')
    cv2.imwrite(save_name, merge_img)
    return colors


if __name__ == '__main__':
    data = np.load(r'C:\Users\ECNU\Desktop\delete_me\20230101_00_400_400.npy')[-1]
    data[data < 0.1] = np.nan
    make_fig(data, area=((40, 120, 20), (20, 60, 20)))
