import os.path

import numpy as np
from einops import rearrange
import rasterio
from rasterio.transform import from_origin
import netCDF4 as nc


def dms_to_d10(data):
    d, m, s = float(data[0]), float(data[1]), float(data[2])
    return d + 1 / 60 * m + 1 / 60 / 60 * s


def make_dms(s):
    print(s)
    temp = s.split('°')
    s_d = temp[0]
    temp = temp[1].split('′')
    s_m = temp[0]
    if len(temp) > 1 and temp[1]:
        s_s = temp[1].split('″')[0]
    else:
        s_s = 0
    result = [float(s_d), float(s_m), float(s_s)]
    return result


def save_np2tif(np_data, output_dir, out_name, coordinate=None, resolution=None, dtype=None):
    h, w = np_data.shape
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, out_name)
    if coordinate:
        # 创建带地理坐标信息的 GeoTIFF 文件
        left, top = coordinate
        res1, res2 = resolution
        # 左上角坐标， 先经后纬
        transform = from_origin(left, top, res1, res2)
        with rasterio.open(save_path, "w", driver="GTiff", width=w, height=h, count=1,
                           dtype=dtype if dtype else np_data.dtype, crs="EPSG:4326", transform=transform) as dst:
            dst.write(np_data, 1)
        print(f"GeoTIFF '{save_path}' generated with geographic coordinates.")
    else:
        # 将数据写入TIFF文件
        with rasterio.open(save_path, "w", width=w, height=h, count=1, dtype=np_data.dtype) as dst:
            dst.write(np_data, 1)
        print(f"TIFF image saved as '{save_path}'")


def np2tif(input_data, save_path, out_name='out', left=None, top=None, x_res=None, y_res=None, dtype=None):
    if type(input_data) == str:
        np_data = np.load(input_data)
    else:
        np_data = input_data
    shape = np_data.shape
    if len(shape) < 2:
        raise Exception(str(shape) + 'is less than 2 Dimensions')
    mode_list = ['d' + str(i) for i in range(len(shape))]

    mode_str = ' '.join(mode_list) + '->(' + ' '.join(mode_list[:-2]) + ') ' + mode_list[-2] + ' ' + mode_list[-1]
    np_data = rearrange(np_data, mode_str)
    if left is not None and top is not None:
        coordinate = (left, top)
    else:
        coordinate = None
    for idx, single_np in enumerate(np_data):
        name = ''
        idx_tmp = idx
        for s in range(len(shape[:-2])):
            temp = idx_tmp // np.prod(shape[s + 1:-2], axis=None)
            name += str(int(temp)) + '-'
            idx_tmp -= temp * np.prod(shape[s + 1:-2], axis=None)
        name = name + out_name + '.tif'
        save_np2tif(single_np, save_path, name, coordinate=coordinate, resolution=(x_res, y_res), dtype=dtype)


def nc2np(input_data):
    if type(input_data) == str:
        nc_data = nc.Dataset(input_data)  # 读取.nc文件，传入f中。此时f包含了该.nc文件的全部信息
    else:
        nc_data = input_data
    vars = []
    max_shape = 0
    for name, var in nc_data.variables.items():
        if len(var.shape) > max_shape:
            max_shape = len(var.shape)
            vars = [name]
        elif len(var.shape) == max_shape:
            vars.append(name)
    np_data = []
    for var in vars:
        np_data.append(np.array(nc_data[var][:]))
    np_data = np.array(np_data)
    return np_data
