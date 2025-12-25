import random

import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from jacksung.utils.data_convert import Coordinate
import os
from importlib import resources
from tqdm import tqdm


def get_pixel_by_coord(img, coord, x, y):
    left, top, x_res, y_res = coord.left, coord.top, coord.x_res, coord.y_res
    if x < left or y > top:
        raise Exception(f'x:{x} or y:{y} is lower than border {left},{top}!'
                        f'left:{left}, top:{top},x_res:{x_res}, y_res:{y_res}.')
    s = img.shape
    if x > left + s[-1] * x_res or y < top - s[-2] * y_res:
        raise Exception(f'x:{x} or y:{y} is greater than border {left + s[-1] * x_res},{top - s[-2] * y_res}!'
                        f'left:{left}, top:{top},x_res:{x_res}, y_res:{y_res}.')
    return img[..., int((top - y) // y_res), int((x - left) // x_res)]


def draw_text(img, xy, font=None, font_size=35, text='test text', color=(0, 0, 0)):
    if xy[0] < 0 or xy[1] < 0:
        xy_txt = (0 if xy[0] < 0 else xy[0], 0 if xy[1] < 0 else xy[1])
        h, w, c = img.shape
        white_block = make_block(h, w, color=(255, 255, 255))
        txt_block = _draw_text(white_block, xy_txt, font, font_size, text, color)
        non_white_mask = np.any(txt_block != [255, 255, 255], axis=-1)
        row_indices, col_indices = np.where(non_white_mask)
        # 3. 计算最大/最小索引
        min_row, max_row = row_indices.min(), row_indices.max()
        min_col, max_col = col_indices.min(), col_indices.max()
        txt_length = max_col - min_col
        txt_width = max_row - min_row
        if xy[0] < 0:
            xy = ((w - txt_length) // 2, xy[1])
        if xy[1] < 0:
            xy = (xy[0], (h - txt_width) // 2)
    return _draw_text(img, xy, font, font_size, text, color)


def _draw_text(img, xy, font=None, font_size=35, text='test text', color=(0, 0, 0)):
    if font is None:
        try:
            with resources.path("jacksung.libs", "times.ttf") as font_path:
                font = ImageFont.truetype(str(font_path), size=font_size)  # 指定字体大小
        except:
            print('load times ttf failed, using the default font')
            font = ImageFont.load_default()
    im = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(im)
    draw.text(xy, text, font=font, fill=color)
    image = np.array(im)
    return image


def _check_border(in_n, n):
    if in_n < 0:
        return 0
    if in_n > n:
        return n
    return in_n


def border(img, point1, point2, color=(0, 0, 255), border=5):
    point1_h, point1_w = point1
    point2_h, point2_w = point2
    h, w, _ = img.shape
    img[_check_border(point1_h - border // 2, h):    _check_border(point1_h + border // 2, h),
    _check_border(point1_w - border // 2, w):    _check_border(point2_w + border // 2, w), :] = color
    img[_check_border(point2_h - border // 2, h):    _check_border(point2_h + border // 2, h),
    _check_border(point1_w - border // 2, w):    _check_border(point2_w + border // 2, w), :] = color
    img[_check_border(point1_h - border // 2, h):    _check_border(point2_h + border // 2, h),
    _check_border(point1_w - border // 2, w):    _check_border(point1_w + border // 2, w), :] = color
    img[_check_border(point1_h - border // 2, h):    _check_border(point2_h + border // 2, h),
    _check_border(point2_w - border // 2, w):    _check_border(point2_w + border // 2, w), :] = color

    return img


def make_block(h, w, color=(255, 255, 255), dtype=np.int32):
    return np.array([[color for _ in range(w)] for _ in range(h)], dtype=dtype)


def get_color_position(value, colors):
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


def make_color_map(colors, h, w, unit=''):
    colors_map = np.zeros((h, w, 3), dtype=np.uint8) + 255
    l_margin, r_margin = 300, 200
    w = w - l_margin - r_margin
    for i in range(l_margin, w + l_margin):
        i = i - l_margin
        colors_map[:100, i + l_margin] = get_color_position(i / w, colors)
        if i in [0, w // 2, w - 1]:
            text = str(round((i / w) * (colors[-1][0] - colors[0][0]) + colors[0][0]))
            if i == 0:
                text += unit
            try:
                with resources.path("jacksung.libs", "times.ttf") as font_path:
                    font = ImageFont.truetype(str(font_path), size=150)  # 指定字体大小
            except:
                print('load times ttf failed, using the default font')
                font = ImageFont.load_default()
            colors_map = draw_text(colors_map, (i - 100 + l_margin, 100),
                                   font=font, text=text)
    return colors_map


def crop_png(input_path, left=0, top=0, right=None, bottom=None, right_margin=0, bottom_margin=0):
    # 打开 PNG 图像
    image = Image.open(input_path)
    width, height = image.size
    if right is None:
        right = width
    if bottom is None:
        bottom = height
    right -= right_margin
    bottom -= bottom_margin
    # 对图像进行裁剪
    cropped_image = image.crop((left, top, right, bottom))
    # 保存裁剪后的图像
    cropped_image.save(input_path)


def concatenate_images(imgs, direction="h"):
    # 读取所有的图片
    # 获取图片的宽度和高度
    heights = [img.shape[0] for img in imgs]
    widths = [img.shape[1] for img in imgs]

    if direction == "h":
        # 水平拼接，高度不变，宽度为所有图片宽度之和
        max_height = max(heights)
        total_width = sum(widths)
        if imgs[0].ndim == 2:
            new_img = np.zeros((max_height, total_width), dtype=np.uint8)
        else:
            new_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        x_offset = 0
        for img in imgs:
            new_img[:, x_offset:x_offset + img.shape[1]] = img
            x_offset += img.shape[1]
    elif direction == "v":
        # 垂直拼接，宽度不变，高度为所有图片高度之和
        max_width = max(widths)
        total_height = sum(heights)
        if imgs[0].ndim == 2:
            new_img = np.zeros((total_height, max_width), dtype=np.uint8)
        else:
            new_img = np.zeros((total_height, max_width, 3), dtype=np.uint8)
        y_offset = 0
        for img in imgs:
            new_img[y_offset:y_offset + img.shape[0], :] = img
            y_offset += img.shape[0]
    else:
        raise ValueError("Invalid direction. Please choose 'h' or 'v'.")
    return new_img


def create_gif(images_in, output_path, duration=500, idx=None):
    images = []
    if type(images_in) == str:
        for file_name in sorted(os.listdir(images_in)):
            # if file_name.endswith('.png'):
            file_path = os.path.join(images_in, file_name)
            images.append(Image.open(file_path))
    else:
        for img in images_in:
            images.append(Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)))
    if idx is not None:
        images = images[idx[0]:idx[1]]
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


def zoom_image(image, scale_factor=2):
    # 获取图像的尺寸
    height, width = image.shape[:2]
    # 计算缩放后的尺寸
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    # 使用cv2.resize进行缩放
    zoomed_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
    return zoomed_image


def zoomAndDock(img, zoom_rectangle, docker, scale_factor=2, border=10):
    corp_png = img[zoom_rectangle[1]:zoom_rectangle[1] + zoom_rectangle[3],
               zoom_rectangle[0]:zoom_rectangle[0] + zoom_rectangle[2], :]
    corp_png = zoom_image(corp_png, scale_factor=scale_factor)
    cv2.rectangle(img, (zoom_rectangle[0], zoom_rectangle[1]),
                  (zoom_rectangle[0] + zoom_rectangle[2], zoom_rectangle[1] + zoom_rectangle[3]), (0, 0, 255), border)
    corp_png[:, 0:border, :] = [0, 0, 0]
    corp_png[:, corp_png.shape[1] - border:corp_png.shape[1], :] = [0, 0, 0]
    corp_png[0:border, :, :] = [0, 0, 0]
    corp_png[corp_png.shape[0] - border:corp_png.shape[0], :, :] = [0, 0, 0]
    img[docker[0]:docker[0] + corp_png.shape[0], docker[1]:docker[1] + corp_png.shape[1], :] = corp_png
    return img


def draw_boundary(img, percent=95):
    h, w = img.shape[:2]
    gradient = np.zeros((h, w))
    for i in range(4, h - 4):
        for j in range(4, w - 4):
            gradient[i, j] += abs(img[i + 1, j] - img[i - 1, j])
            gradient[i, j] += abs(img[i + 2, j] - img[i - 2, j])
            gradient[i, j] += abs(img[i + 3, j] - img[i - 3, j])
            gradient[i, j] += abs(img[i + 4, j] - img[i - 4, j])
            gradient[i, j] += abs(img[i, j + 1] - img[i, j - 1])
            gradient[i, j] += abs(img[i, j + 2] - img[i, j - 2])
            gradient[i, j] += abs(img[i, j + 3] - img[i, j - 3])
            gradient[i, j] += abs(img[i, j + 4] - img[i, j - 4])

    threshold = np.percentile(gradient, percent)
    boundary_mask = gradient > threshold
    return boundary_mask


if __name__ == '__main__':
    img1 = np.array(Image.open(r'C:\Users\ECNU\Desktop\f_atn_visu\f-2.tif'))
    img2 = np.array(Image.open(r'C:\Users\ECNU\Desktop\n_atn_visu\n-2.tif'))
    img3 = np.array(Image.open(r'C:\Users\ECNU\Desktop\p_atn_visu\p-2.tif'))
    # img1 = zoom_image(img1, 0.25)
    # img2 = zoom_image(img2, 0.25)
    # img3 = zoom_image(img3, 0.25)
    threshold = 0.6
    img1[img1 > threshold] = 1
    img1[img1 <= threshold] = 0
    img2[img2 > threshold] = 1
    img2[img2 <= threshold] = 0
    img3[img3 > threshold] = 1
    img3[img3 <= threshold] = 0

    boundary1 = draw_boundary(img1, 96)
    boundary2 = draw_boundary(img2, 96)
    boundary3 = draw_boundary(img3, 96)

    img = np.zeros(img1.shape + (3,))
    img[boundary1] = [255, 0, 0]
    img[boundary2] = [0, 255, 0]
    img[boundary3] = [0, 0, 255]
    img[700 - 5:700 + 5, 332 - 5:332 + 5, :] = [255, 255, 255]
    img[700, :, :] = [255, 255, 255]
    img[:, 332, :] = [255, 255, 255]
    for i in range(4):
        lon_att = np.array(Image.open(rf'C:\Users\ECNU\Desktop\lon_atn_visu\lon_atn-{i}.tif'))
        lat_att = np.array(Image.open(rf'C:\Users\ECNU\Desktop\lat_atn_visu\lat_atn-{i}.tif'))
        lon_att = zoom_image(lon_att, 4)[332]
        lat_att = zoom_image(lat_att, 4)[700]
        lon_att = (lon_att - np.min(lon_att)) / (np.percentile(lon_att, 90) - np.min(lon_att)) * 255
        lat_att = (lat_att - np.min(lat_att)) / (np.percentile(lat_att, 90) - np.min(lat_att)) * 255
        # att = np.zeros((200, 200))
        img[700 - 2:700 + 2, :] = np.repeat(np.repeat(lon_att[np.newaxis, :, np.newaxis], 3, axis=2), 4, axis=0)
        img[:, 332 - 2:332 + 2] = np.repeat(np.repeat(lat_att[:, np.newaxis, np.newaxis], 3, axis=2), 4, axis=1)
        cv2.imwrite(rf'C:\Users\ECNU\Desktop\f_atn_visu\boundary{i}.png', zoom_image(img, 4).astype(np.uint8))

        # cv2.imwrite(rf'C:\Users\ECNU\Desktop\f_atn_visu\att.png', att.astype(np.uint8))
        # cv2.waitKey()
