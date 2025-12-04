import os
from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
from einops import rearrange, repeat
from jacksung.utils.data_convert import np2tif, get_transform_from_lonlat_matrices, Coordinate
import xarray as xr
from pyresample import create_area_def, kd_tree
from pyresample.geometry import AreaDefinition
from jacksung.utils.time import Stopwatch
from jacksung.utils.cache import Cache


def get_resample_infos(hdf_path, lock=None, cache=None):
    if lock:
        lock.acquire()
    ds = nc.Dataset(hdf_path)
    if lock:
        lock.release()
    ld = float(ds['nominal_satellite_subpoint_lon'][:])
    ld = round(ld, 2)
    if cache:
        cache_result = cache.get_key_in_cache(ld)
        if cache_result is not None:
            return cache_result
    # 原始GEOS投影
    goes_proj_str = ds['goes_imager_projection']
    h = float(goes_proj_str.perspective_point_height)
    x = ds['x'][:] * h  # 投影x坐标 (radians)
    y = ds['y'][:] * h  # 投影y坐标 (radians)
    # 计算投影范围 (根据x, y的边界)
    half_pixel_width = (x[1] - x[0]) / 2.0
    half_pixel_height = (y[1] - y[0]) / 2.0
    area_extent = (x[0] - half_pixel_width, y[-1] - half_pixel_height,
                   x[-1] + half_pixel_width, y[0] + half_pixel_height)
    goes_area = AreaDefinition(
        area_id='goes_fixed_grid', proj_id='goes_geos', description='GOES Fixed Grid',
        projection={
            'proj': 'geos',
            'lon_0': goes_proj_str.longitude_of_projection_origin,
            'h': goes_proj_str.perspective_point_height,
            'x_0': 0,
            'y_0': 0,
            'a': goes_proj_str.semi_major_axis,
            'b': goes_proj_str.semi_minor_axis,
            'sweep': goes_proj_str.sweep_angle_axis
        },
        width=len(x), height=len(y), area_extent=area_extent)
    left = ld - 60
    right = ld + 60
    target_areas = []
    if left < -180:
        # 跨越180度经线，分两部分重采样
        # 左半部分
        target_area_left = create_area_def(
            area_id='wgs84_left', projection='EPSG:4326', area_extent=[left + 360, -60, 180, 60],
            resolution=(0.05, 0.05), units='degrees')
        target_areas.append(target_area_left)
        # 右半部分
        target_area_right = create_area_def(
            area_id='wgs84_right', projection='EPSG:4326', area_extent=[-180, -60, right, 60],
            resolution=(0.05, 0.05), units='degrees')
        target_areas.append(target_area_right)
    else:
        target_area = create_area_def(
            area_id='wgs84', projection='EPSG:4326', area_extent=[left, -60, right, 60],
            resolution=(0.05, 0.05), units='degrees')
        target_areas.append(target_area)
    resample_infos = []
    for target_area in target_areas:
        # 使用最近邻法重采样，对于分类数据；对于连续数据，可以使用 ‘bilinear’
        resample_infos.append(
            kd_tree.get_neighbour_info(goes_area, target_area, radius_of_influence=5000, neighbours=1))
    if cache:
        cache.add_key(ld, resample_infos)
    return resample_infos


def getSingleChannelNPfromHDF(hdf_path, lock=None, return_coord=False, only_coord=False, resample_infos=None):
    if lock:
        lock.acquire()
    ds = nc.Dataset(hdf_path)
    if lock:
        lock.release()
    ld = float(ds['nominal_satellite_subpoint_lon'][:])
    np_data = np.array(ds['Rad'][:]).astype(np.float32)
    ld = round(ld, 2)
    left = ld - 60
    right = ld + 60
    coord = Coordinate(left=left, bottom=-60, right=right, top=60, x_res=0.05, y_res=0.05)
    if only_coord:
        return coord
    np_datas = []
    if resample_infos is None:
        resample_infos = get_resample_infos(hdf_path, lock=lock)
    for info in resample_infos:
        valid_input_index, valid_output_index, index_array, distance_array = info
        # 使用最近邻法重采样，对于分类数据；对于连续数据，可以使用 ‘bilinear’
        results = kd_tree.get_sample_from_neighbour_info(
            'nn', output_shape=(coord.h, int(len(info[1]) / 2400)), data=np_data, valid_input_index=valid_input_index,
            valid_output_index=valid_output_index, index_array=index_array, fill_value=np.nan)
        np_datas.append(results)
    # 合并两部分数据
    np_data = np.concatenate(np_datas, axis=1)
    if return_coord:
        return np_data, coord
    else:
        return np_data


def get_filename_by_date_from_dir(dir_path, date, satellite='G18'):
    file_lists = {}
    for file in os.listdir(dir_path):
        if not file.endswith('.nc'):
            continue
        splits = file.split('_')
        year = int(splits[3][1:5])
        doy = int(splits[3][5:8])
        hour = int(splits[3][8:10])
        minute = int(splits[3][10:12])
        file_date = datetime(year=year, month=1, day=1) + timedelta(days=doy - 1, hours=hour, minutes=minute)
        if date == file_date and splits[2] == satellite:
            file_lists[int(splits[1].split('-')[3][3:])] = file
    return file_lists


def getNPfromDir(dir_path, date, satellite='G18', lock=None, return_coord=False, infos=None, cache=None):
    np_data = None
    coord = None
    data_channel_count = 0
    files = get_filename_by_date_from_dir(dir_path, date, satellite)
    for channel, file in files.items():
        if infos is None:
            infos = get_resample_infos(os.path.join(dir_path, file), lock=lock, cache=cache)
        channel_data, coord = getSingleChannelNPfromHDF(
            os.path.join(dir_path, file), return_coord=True, resample_infos=infos)
        if channel_data is None:
            raise Exception(f"文件{file}，通道 {channel} 数据获取失败")
        if np_data is None:
            np_data = np.full([9] + list(channel_data.shape), np.nan)
        np_data[channel - 8] = channel_data
        data_channel_count += 1
    if data_channel_count < 9:
        raise Exception(
            f"文件夹{dir_path}中，卫星 {satellite} 在时间 {date} 的数据通道不完整，仅获取到 {data_channel_count} 个通道")
    if return_coord:
        return np_data, coord
    else:
        return np_data


if __name__ == '__main__':
    np_data = getNPfromDir(rf'D:\python_Project\Huayu_Global\file_download\2022\12\30',
                           datetime(year=2022, month=12, day=30, hour=3, minute=0), satellite='G18')
    np2tif(np_data, save_path='test_goes', out_name='GOES18',
           left=-137 - 60, top=60, x_res=0.05, y_res=0.05, dtype=np.float32)
    np_data = getNPfromDir(rf'D:\python_Project\Huayu_Global\file_download\2022\12\30',
                           datetime(year=2022, month=12, day=30, hour=3, minute=0), satellite='G16')
    np2tif(np_data, save_path='test_goes', out_name='GOES16',
           left=-75.2 - 60, top=60, x_res=0.05, y_res=0.05, dtype=np.float32)
