import netCDF4 as nc
import numpy as np
from einops import rearrange, repeat
from jacksung.utils.data_convert import np2tif, get_transform_from_lonlat_matrices


def getNPfromHDF(hdf_path, lock=None, save_file=True):
    if lock:
        lock.acquire()
    ds = nc.Dataset(hdf_path)
    if lock:
        lock.release()
    np_data = np.array(ds['hourlyPrecipRateGC'][:]).astype(np.float32)[0]
    ds.close()
    np_data[np_data < 0] = np.nan
    np_data = np_data[::-1, :]
    if save_file:
        np2tif(np_data, save_path='np2tif_dir', out_name='gsmap', dtype='float32',
               left=-180, top=90, x_res=0.1, y_res=0.1)
    return np_data


if __name__ == '__main__':
    getNPfromHDF(rf'D:\python_Project\Huayu_Global\file_download\gsmap_now_rain.20220702.0300.nc')
