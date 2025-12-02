import os
from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
from einops import rearrange, repeat
from jacksung.utils.data_convert import np2tif, get_transform_from_lonlat_matrices, Coordinate
import xarray as xr
from pyresample import create_area_def, kd_tree
from pyresample.geometry import AreaDefinition
import cartopy.crs as ccrs


def getSingleChannelNPfromHDF(hdf_path, lock=None, print_log=False, return_coord=False):
    if lock:
        lock.acquire()
    ds = nc.Dataset(hdf_path)
    if lock:
        lock.release()
    ld = float(ds['nominal_satellite_subpoint_lon'][:])
    np_data = np.array(ds['Rad'][:]).astype(np.float32)
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
        area_id='goes_fixed_grid',
        proj_id='goes_geos',  # 新增的必需参数
        description='GOES Fixed Grid',
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
        width=len(x),
        height=len(y),
        area_extent=area_extent
    )
    left = ld - 60
    right = ld + 60
    print('goes subpoint:' + str(ld))
    coord = Coordinate(left=left, bottom=-60, right=right, top=60, x_res=0.05, y_res=0.05)
    np_datas = []
    target_areas = []
    if left < -180:
        # 跨越180度经线，分两部分重采样
        # 左半部分
        target_area_left = create_area_def(
            area_id='wgs84_left',
            projection='EPSG:4326',  # WGS84的EPSG代码
            area_extent=[left + 360, -60, 180, 60],
            resolution=(0.05, 0.05),  # 分辨率，例如 0.05度 (约5.5km)
            units='degrees'
        )
        target_areas.append(target_area_left)
        # 右半部分
        target_area_right = create_area_def(
            area_id='wgs84_right',
            projection='EPSG:4326',  # WGS84的EPSG代码
            area_extent=[-180, -60, right, 60],
            resolution=(0.05, 0.05),  # 分辨率，例如 0.05度 (约5.5km)
            units='degrees'
        )
        target_areas.append(target_area_right)
    else:
        target_area = create_area_def(
            area_id='wgs84',
            projection='EPSG:4326',  # WGS84的EPSG代码
            area_extent=[left, -60, right, 60],
            resolution=(0.05, 0.05),  # 分辨率，例如 0.05度 (约5.5km)
            units='degrees'
        )
        target_areas.append(target_area)

    for target_area in target_areas:
        # 使用最近邻法重采样，对于分类数据；对于连续数据，可以使用 ‘bilinear’
        results = kd_tree.resample_nearest(
            goes_area,
            np_data,  # 原始数据数组
            target_area,
            radius_of_influence=50000,  # 搜索半径 (米)，对于高分辨率数据可能需要调整
            fill_value=np.nan  # 无数据区域填充值
        )
        np_datas.append(results)
    # 合并两部分数据
    np_data = np.concatenate(np_datas, axis=1)
    if return_coord:
        return np_data, coord
    else:
        return np_data


def getNPfromDir(dir_path, date, satellite='G18', lock=None, return_coord=False):
    np_data = None
    coord = None
    for file in os.listdir(dir_path):
        splits = file.split('_')
        year = int(splits[3][1:5])
        doy = int(splits[3][5:8])
        hour = int(splits[3][8:10])
        minute = int(splits[3][10:12])
        file_date = datetime(year=year, month=1, day=1) + timedelta(days=doy - 1, hours=hour, minutes=minute)
        if date == file_date and splits[2] == satellite:
            channel = int(splits[1].split('-')[3][3:])
            channel_data, coord = getSingleChannelNPfromHDF(os.path.join(dir_path, file), return_coord=True)
            if channel_data is None:
                raise Exception(f"文件{file}，通道 {channel} 数据获取失败")
            if np_data is None:
                np_data = np.full([9] + list(channel_data.shape), np.nan)
            np_data[channel - 8] = channel_data
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
