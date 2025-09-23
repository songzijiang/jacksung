import sys
import threading

sys.path.append('../')
import shutil
from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import os
from jacksung.utils.data_convert import nc2np, np2tif
from jacksung.utils.image import crop_png, zoom_image, zoomAndDock
from jacksung.utils.cache import Cache
import rasterio
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import netCDF4 as nc
from jacksung.utils.data_convert import np2tif, Coordinate
from tqdm import tqdm
from jacksung.utils.multi_task import MultiTasks, type_process
import cv2

reference_cache = Cache(10)
x_range = {'left': -60, 'top': 60, 'bottom': -60, 'right': 60, 'width': 2400, 'height': 2400}
min_x_range = {'left': -60, 'top': 60, 'bottom': -60, 'right': 60, 'width': 480, 'height': 480}
static_params = {4000: {'l': 2747, 'c': 2747, 'COFF': 1373.5, 'CFAC': 10233137, 'LOFF': 1373.5, 'LFAC': 10233137},
                 2000: {'l': 5495, 'c': 5495, 'COFF': 2747.5, 'CFAC': 20466274, 'LOFF': 2747.5, 'LFAC': 20466274},
                 1000: {'l': 10991, 'c': 10991, 'COFF': 5495.5, 'CFAC': 40932549, 'LOFF': 5495.5, 'LFAC': 40932549},
                 500: {'l': 21983, 'c': 21983, 'COFF': 10991.5, 'CFAC': 81865099, 'LOFF': 10991.5, 'LFAC': 81865099},
                 250: {'l': 43967, 'c': 43967, 'COFF': 21983.5, 'CFAC': 163730199, 'LOFF': 21983.5, 'LFAC': 163730199}}


def getFY_coord(ld):
    return Coordinate(left=ld + x_range['left'], top=x_range['top'], right=ld + x_range['right'],
                      bottom=x_range['bottom'], h=x_range['height'], w=x_range['width'])
    # return Coordinate(left=ld - 45, top=36, right=ld + 45, bottom=-36, h=1571, w=1963)


def getFY_coord_min(ld):
    return Coordinate(left=ld + min_x_range['left'], top=min_x_range['top'], right=ld + min_x_range['right'],
                      bottom=min_x_range['bottom'], h=min_x_range['height'], w=min_x_range['width'])


def getFY_coord_clip(area=((100, 140, 10), (20, 60, 10))):
    return Coordinate(left=area[0][0], top=area[1][1], right=area[0][1], bottom=area[1][0], h=800, w=800)


# FY4星下点行列号转经纬度
def xy2coordinate(l, c, ld=105, res=4000):
    ea = 6378.137
    eb = 6356.7523
    h = 42164
    # 4000m分辨率
    COFF = static_params[res]['COFF']
    CFAC = static_params[res]['CFAC']
    LOFF = static_params[res]['LOFF']
    LFAC = static_params[res]['LFAC']

    x = (np.pi * (c - COFF)) / (180 * (2 ** -16) * CFAC)
    y = (np.pi * (l - LOFF)) / (180 * (2 ** -16) * LFAC)

    sd_t1 = np.square(h * np.cos(x) * np.cos(y))
    sd_t2 = np.square(np.cos(y)) + np.square(ea) / np.square(eb) * np.square(np.sin(y))
    sd_t3 = np.square(h) - np.square(ea)
    sd = np.sqrt(sd_t1 - sd_t2 * sd_t3)
    sn = (h * np.cos(x) * np.cos(y) - sd) / (np.cos(y) ** 2 + ea ** 2 / eb ** 2 * np.sin(y) ** 2)

    s1 = h - sn * np.cos(x) * np.cos(y)
    s2 = sn * np.sin(x) * np.cos(y)
    s3 = -sn * np.sin(y)
    sxy = np.sqrt(s1 ** 2 + s2 ** 2)

    lon = 180 / np.pi * np.arctan(s2 / s1) + ld
    lat = 180 / np.pi * np.arctan(ea ** 2 / eb ** 2 * s3 / sxy)
    return lon, lat


def convert_file2idx(file_name):
    file_name = file_name.replace('.npy', '')
    h, m = file_name[:2], file_name[2:4]
    return int(h) * 4 + int(m) // 15


def get_reference(ld):
    # 构造控制点列表 gcps_list
    gcps_list = []
    step = 50
    last_corrd = []
    lc_list = []
    latlon_list = []
    for l in range(0, 2748, step):
        for c in range(0, 2748, step):
            lon, lat = xy2coordinate(l, c, ld=ld)
            if str(lon) == 'nan' or str(lat) == 'nan':
                continue
            skip_flag = False
            for corrd in last_corrd:
                if (corrd[0] - lon) ** 2 + (corrd[1] - lat) ** 2 <= 100:
                    skip_flag = True
                    break
            if skip_flag:
                continue
            last_corrd.append([lon, lat])
            gcps_list.append(gdal.GCP(lon, lat, 0, c, l))
            lc_list.append((l, c))
            latlon_list.append((lon, lat))
    # 设置空间参考
    # print('控制点数目：', len(gcps_list))
    # print([(l, c) for l, c in lc_list])
    # print([(lon, lat) for lon, lat in latlon_list])
    spatial_reference = osr.SpatialReference()
    spatial_reference.SetWellKnownGeogCS('WGS84')
    return spatial_reference, gcps_list


def getNPfromHDFClip(ld, file_path, file_type='FDI', lock=None, area=((100, 140, 10), (20, 60, 10))):
    lon_d = int((ld - (area[0][0] + area[0][1]) / 2) * 20)
    lat_d = int(((area[1][0] + area[1][1]) / 2) * 20)
    np_data = getNPfromHDF(file_path, file_type, lock)
    np_data = np_data[:, 800 - lat_d:1600 - lat_d, 800 - lon_d:1600 - lon_d]
    return np_data


def getNPfromHDF(hdf_path, file_type='FDI', lock=None):
    file_name = hdf_path.split(os.sep)[-1]
    file_info = prase_filename(file_name)
    if lock:
        lock.acquire()
    try:
        ds = nc.Dataset(hdf_path)
    except:
        print(f'open {hdf_path} failed')
    if lock:
        lock.release()
    if file_type == 'FDI':
        f = ds.groups['Data']
        np_data = np.zeros((15, 2748, 2748), dtype=np.float32)
        for i in range(1, 16):
            s_i = '0' + str(i) if i < 10 else str(i)
            data = np.array(f[f'NOMChannel{s_i}'][:]).astype(np.float32)
            data[data > 10000] = np.nan
            np_data[i - 1] = data
        # np_data = np_data[6:15]
        in_out_idx = [6, 15]
    elif file_type == 'QPE':
        np_data = np.array(ds['Precipitation'][:]).astype(np.float32)
        np_data = np_data[np.newaxis, :]
        in_out_idx = [0, 1]
    else:
        np_data = None
        in_out_idx = None
        raise Exception(rf'file_type {file_type} err')
    ds.close()
    r = reference_cache.get_key_in_cache(file_info['position'])
    if r is None:
        print(f'get reference of {file_info["position"]}')
        r = reference_cache.add_key(file_info['position'], get_reference(ld=file_info['position']))

    np_data = _getNPfromHDF_worker(np_data, file_info['start'], r=r, ld=file_info['position'], to_file=False,
                                   in_out_idx=in_out_idx)
    return np_data


def _getNPfromHDF_worker(read_np_data, current_date, data_dir=None, ld=None, r=None, to_file=True, in_out_idx=(6, 15)):
    tmp_dir = 'make_temp' + str(randint(10000000, 99999999))
    file = current_date.strftime("%H%M") + '.npy'
    os.makedirs(tmp_dir, exist_ok=True)
    save_path = f'{current_date.year}{current_date.month}{current_date.day}{file.replace(".npy", "")}'
    np2tif(read_np_data, tmp_dir, save_path, print_log=False)
    in_idx, out_idx = in_out_idx
    np_data = np.zeros((out_idx - in_idx, x_range['height'], x_range['width']), dtype=np.float16)
    for i in range(in_idx, out_idx):
        out_path = f'{tmp_dir}/{save_path}-{i}-ctrl.tif'
        registration(f'{tmp_dir}/{save_path}-{i}.tif', out_path, ld, *r)
        img = cv2.imread(out_path, -1)
        if np.isnan(img).any():
            shutil.rmtree(tmp_dir)
            return None
        np_data[i - in_idx] = img.astype(np.float16)
    # raise Exception('manual stop')
    shutil.rmtree(tmp_dir)
    if to_file:
        os.makedirs(f'{data_dir}/dataset/{current_date.year}/{current_date.month}/{current_date.day}', exist_ok=True)
        np.save(
            f'{data_dir}/dataset/{current_date.year}/{current_date.month}/{current_date.day}/{convert_file2idx(file)}',
            np_data)
    else:
        return np_data


# 解析FY文件的文件名
def prase_filename(filename):
    m_list = filename.replace('.HDF', '').split('_')
    # FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250606171500_20250606172959_4000M_V0001.HDF
    return {'satellite': m_list[0], 'sensor': m_list[1], 'area': m_list[3], 'position': int(m_list[4][:3]),
            'file_level': m_list[5], 'data_name': m_list[6], 'start': datetime.strptime(m_list[9], '%Y%m%d%H%M%S'),
            'end': datetime.strptime(m_list[10], '%Y%m%d%H%M%S'), 'resolution': m_list[11]}


def registration(input_path, out_path, ld, spatial_reference, gcps_list):
    """
    基于python GDAL配准
    :param input_path: 需要配准的栅格文件
    :param out_path: 输出配准后的栅格文件位置
    :param top_left: 左上角坐标
    :param bottom_right: 右下角坐标
    :param ik: 行空白分辨率
    :param jk: 列空白分辨率
    :return:
    """
    # 打开栅格文件
    dataset = gdal.Open(input_path, gdal.GA_Update)
    # 添加控制点
    dataset.SetGCPs(gcps_list, spatial_reference.ExportToWkt())
    # tps校正 重采样:最邻近法
    gdal.Warp(out_path, dataset,
              format='GTiff',
              outputBounds=[ld + x_range['left'], x_range['bottom'], ld + x_range['right'], x_range['top']],
              resampleAlg=gdal.GRIORA_NearestNeighbour,
              width=x_range['width'],
              height=x_range['height'],
              tps=True,
              dstSRS='EPSG:4326')


def _prase_nc_worker(root_path, target_dir, file, lock=None):
    ps = prase_filename(file)
    file_name = ps["start"].strftime("%H%M")
    hdf_path = f'{root_path}/{target_dir}/{ps["start"].year}/{ps["start"].month}/{ps["start"].day}/{file}'
    save_path = f'{root_path}/npy/{ps["start"].year}/{ps["start"].month}/{ps["start"].day}'
    if not os.path.exists(hdf_path):
        tqdm.write(f'## {ps["start"].strftime("%Y%m%d %H:%M")} None ##')
        return False
    if os.path.exists(f'{save_path}/{convert_file2idx(file_name)}.npy'):
        tqdm.write(f'## {ps["start"].strftime("%Y%m%d %H:%M")} already exist ##')
        return True
    try:
        n_data = getNPfromHDF(hdf_path, lock=lock)
        os.makedirs(save_path, exist_ok=True)
        if n_data is None:
            tqdm.write(f'## {ps["start"].strftime("%Y%m%d %H:%M")} None ##')
            return False
            # raise Exception(f'nan in cliped {hdf_path}')
        np.save(f'{save_path}/{convert_file2idx(file_name)}', n_data)
    except Exception as e:
        # raise e
        tqdm.write(f'## {ps["start"].strftime("%Y%m%d %H:%M")} err ##')
        return False
    tqdm.write(f'{ps["start"].strftime("%Y%m%d %H:%M")} down')
    return True


# 把FY4的HDF文件转为npy文件
def make_fynp(root_path, target_dir, time_set):
    mt = MultiTasks(40)
    for current_date in time_set:
        for file in os.listdir(
                f'{root_path}/{target_dir}/{current_date.year}/{current_date.month}/{current_date.day}'):
            if file.endswith('.HDF'):
                mt.add_task(file, _prase_nc_worker, [root_path, target_dir, file, mt.thread_mutex])
                # _prase_nc_worker(root_path, target_dir, file, None)
    err_list = []
    for key, flag in mt.execute_task().items():
        if not flag:
            err_list.append(key + '\n')
    with open(f'err.log', 'w') as f:
        f.writelines(err_list)
    print(f'all done, {len(err_list)} in err.log')


def get_ld(data):
    if data < datetime(2024, 3, 1):
        return 133
    else:
        return 105


# 根据日期获取星下点位置
def get_filename_by_date(file_date):
    ld = get_ld(file_date)
    filename = rf'FY4B-_AGRI--_N_DISK_{ld}E_L1-_FDI-_MULT_NOM_{file_date.strftime("%Y%m%d%H%M%S")}_{(file_date + timedelta(minutes=14, seconds=59)).strftime("%Y%m%d%H%M%S")}_4000M_V0001.HDF'
    return filename
