import os.path

import numpy as np
from einops import rearrange
import rasterio
from rasterio.transform import from_origin
import netCDF4 as nc


class Coordinate:
    def __init__(self, left, top, x_res=None, y_res=None, right=None, bottom=None, h=None, w=None):
        self.left = left
        self.top = top
        self.x_res = x_res
        self.y_res = y_res
        self.h = h
        self.w = w
        self.right = right
        self.bottom = bottom
        if x_res is None and right is not None and w is not None:
            self.x_res = (right - left) / w
        if y_res is None and bottom is not None and h is not None:
            self.y_res = (top - bottom) / h
        if self.x_res is None or self.y_res is None or self.left is None or self.top is None:
            raise Exception(f'None parameter x_res, y_res, left, top: {self.x_res},{self.y_res},{self.left},{self.top}')


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


def _save_np2tif(np_data, output_dir, out_name, coordinate=None, resolution=None, dtype=None, print_log=False):
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
        if print_log:
            print(f"GeoTIFF '{save_path}' generated with geographic coordinates.")
    else:
        # 将数据写入TIFF文件
        with rasterio.open(save_path, "w", width=w, height=h, count=1, dtype=dtype if dtype else np_data.dtype) as dst:
            dst.write(np_data, 1)
        if print_log:
            print(f"TIFF image saved as '{save_path}'")


def np2tif(input_data, save_path='np2tif_dir', out_name='', left=None, top=None, x_res=None, y_res=None, dtype=None,
           dim_value=None, coord=None, print_log=True):
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
    elif coord is not None:
        coordinate = (coord.left, coord.top)
        x_res, y_res = coord.x_res, coord.y_res
    else:
        coordinate = None
    for idx, single_np in enumerate(np_data):
        name = ''
        idx_tmp = idx
        for s in range(len(shape[:-2])):
            name += '-'
            temp = int(idx_tmp // np.prod(shape[s + 1:-2], axis=None))
            if dim_value is not None:
                plus_name = str(dim_value[s]['value'][temp])
            else:
                plus_name = str(temp)
            name += plus_name
            idx_tmp -= temp * np.prod(shape[s + 1:-2], axis=None)
        name = out_name + name + '.tif'
        _save_np2tif(single_np, save_path, name, coordinate=coordinate, resolution=(x_res, y_res), dtype=dtype,
                     print_log=print_log)


def nc2tif(input_data, save_path='np2tif_dir', lock=None):
    np_data, dim_value = nc2np(input_data, lock)
    np2tif(np_data, save_path, dim_value=dim_value)


def nc2np(input_data, lock=None, return_dim=True):
    if type(input_data) == str:
        if lock:
            lock.acquire()
        nc_data = nc.Dataset(input_data)  # 读取.nc文件，传入f中。此时f包含了该.nc文件的全部信息
        if lock:
            lock.release()
    else:
        nc_data = input_data
    vars = []
    max_shape = 0
    dimensions = {}
    for name, var in nc_data.variables.items():
        if len(var.shape) > max_shape:
            max_shape = len(var.shape)
            vars = [name]
        elif len(var.shape) == max_shape:
            vars.append(name)
        if len(var.shape) == 1:
            dimensions[name] = list(nc_data[name][:])
    np_data = []
    for var in vars:
        np_data.append(np.array(nc_data[var][:]))
    np_data = np.array(np_data)
    value_idx = 0
    while 'value' + str(value_idx) in dimensions:
        value_idx += 1
    value_key = 'value' + str(value_idx)
    dimensions[value_key] = vars
    if return_dim:
        np_idx = [value_key] + list(nc_data[var].dimensions)
        dim_value = [{'dim_name': key, 'value': dimensions[key]} for key in np_idx]
    else:
        dim_value = None
    if type(input_data) == str:
        nc_data.close()
    return np_data, dim_value


def add_None(a, b):
    if a is None:
        return b
    else:
        return a + b


if __name__ == '__main__':
    np_data, dim = nc2np(r'C:\Users\jackSung\Desktop\download.nc')
    np2tif(np_data, 'com', dim_value=dim)
    print(dim)
