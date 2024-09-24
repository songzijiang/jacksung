import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from jacksung.utils.data_convert import Coordinate
import os


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


def draw_text(img, xy, font=ImageFont.truetype(r'times.ttf', 35), text='test text', color=(0, 0, 0)):
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


if __name__ == '__main__':
    path = r'D:\python_Project\FYpredict\metrics\make_figure\band_metrics.png'
    crop_png(path, right_margin=50)
