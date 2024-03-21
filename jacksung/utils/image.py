import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np


def draw_text(img, xy, font, text):
    # font = ImageFont.truetype(r'arial.ttf', 35)
    im = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(im)
    draw.text(xy, text, font=font, fill=(0, 0, 0))
    image = np.array(im)
    return image


def make_block(h, w, color=(255, 255, 255)):
    return np.array([[color for _ in range(w)] for _ in range(h)])


def crop_png(input_path, left, top, right, bottom):
    # 打开 PNG 图像
    image = Image.open(input_path)
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
        new_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        x_offset = 0
        for img in imgs:
            new_img[:, x_offset:x_offset + img.shape[1]] = img
            x_offset += img.shape[1]
    elif direction == "v":
        # 垂直拼接，宽度不变，高度为所有图片高度之和
        max_width = max(widths)
        total_height = sum(heights)
        new_img = np.zeros((total_height, max_width, 3), dtype=np.uint8)
        y_offset = 0
        for img in imgs:
            new_img[y_offset:y_offset + img.shape[0], :] = img
            y_offset += img.shape[0]
    else:
        raise ValueError("Invalid direction. Please choose 'h' or 'v'.")
    return new_img
