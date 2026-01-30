import os
from jacksung.utils.data_convert import np2tif, Coordinate
import netCDF4 as nc
import numpy as np
from datetime import datetime


def getNPfromHDF(hdf_path, lock=None):
    prase_data = prase_filename(os.path.basename(hdf_path))
    ld = prase_data['position']
    coord = Coordinate(left=ld - 60, top=60, right=ld + 60, bottom=-60, x_res=0.05, y_res=0.05)

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
    f = ds.groups['Data']
    np_data = np.zeros((15, 2748, 2748), dtype=np.float32)
    for i in range(1, 16):
        s_i = '0' + str(i) if i < 10 else str(i)
        data = np.array(f[f'NOMChannel{s_i}'][:]).astype(np.float32)
        data[data > 10000] = np.nan
        np_data[i - 1] = data
    # np_data = np_data[6:15]

    in_out_idx = [6, 15]
    ds.close()
    r = get_reference(ld=file_info['position'])

    np_data = _getNPfromHDF_worker(np_data, file_info['start'], r=r, ld=file_info['position'], to_file=False,
                                   in_out_idx=in_out_idx)


def prase_filename(filename):
    m_list = filename.replace('.HDF', '').split('_')
    # FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20250606171500_20250606172959_4000M_V0001.HDF
    # FY3G_PMR--_ORBD_L1_20260127_0209_5000M_V1.HDF
    return {'satellite': m_list[0], 'sensor': m_list[1], 'file_level': m_list[3], 'data_name': m_list[6],
            'date': datetime.strptime(m_list[4] + m_list[5], '%Y%m%d%H%M'), 'resolution': m_list[6]}


if __name__ == '__main__':
    getNPfromHDF(rf'FY3G_PMR--_ORBD_L1_20260127_0209_5000M_V1.HDF')
