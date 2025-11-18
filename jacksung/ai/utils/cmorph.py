import netCDF4 as nc
import numpy as np
from einops import rearrange, repeat
from jacksung.utils.data_convert import np2tif, get_transform_from_lonlat_matrices


def getNPfromHDF(hdf_path, lock=None, save_file=True, print_log=False):
    if lock:
        lock.acquire()
    ds = nc.Dataset(hdf_path)
    if lock:
        lock.release()
    np_data = np.array(ds['cmorph'][:]).astype(np.float32)
    lon_array = np.array(ds['lon'][:]).astype(np.float32)
    lat_array = np.array(ds['lat'][:]).astype(np.float32)
    lon_dim = len(lon_array)
    lat_dim = len(lat_array)
    lon_array = repeat(lon_array, 'w -> w h', h=lat_dim)
    lat_array = repeat(lat_array, 'h -> w h', w=lon_dim)
    ds.close()
    # np_data = rearrange(np_data[0], 'w h->h w')[::-1, :]
    np_data[np_data < 0] = 0
    # np_data = np_data[0] + np_data[1]
    transform, avg_error = get_transform_from_lonlat_matrices(
        lon_array=lon_array,
        lat_array=lat_array,
        gcp_density=20,  # 范围越大，gcp_density建议越大
        print_log=print_log
    )
    if save_file:
        np2tif(np_data, save_path='np2tif_dir', out_name='CMORPH', dtype='float32', transform=transform)
    return np_data, transform


if __name__ == '__main__':
    data = getNPfromHDF(rf'C:\Users\ECNU\PycharmProjects\CMORPH_V1.0_ADJ_8km-30min_2022070203.nc')
    # from datetime import datetime
    #
    # da = datetime.utcfromtimestamp(1656730800)
    # print(da)
    # da = datetime.utcfromtimestamp(1656732600)
    # print(da)
