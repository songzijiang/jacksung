from satpy import Scene
from pyresample import create_area_def
import numpy as np
import os
from datetime import timedelta
from jacksung.utils.data_convert import np2tif, Coordinate
from contextlib import contextmanager


@contextmanager
def satpy_scene_context(filenames, reader):
    """satpy Scene上下文管理器"""
    scn = None
    try:
        scn = Scene(filenames=filenames, reader=reader)
        yield scn
    finally:
        # 显式清理satpy资源
        if scn is not None:
            try:
                # 尝试调用satpy的清理方法
                if hasattr(scn, 'close'):
                    scn.close()
                # 手动删除引用
                del scn
            except Exception as e:
                print(f"清理satpy资源时警告: {e}")


def _define_wgs84_area(resolution=0.05, area_extent=(-14.5, -60, 105.5, 60.0)):
    """定义WGS84坐标系目标区域（60°N-60°S，全经度），保持您原有的区域定义"""

    wgs84_proj = "+proj=longlat +datum=WGS84 +ellps=WGS84 +no_defs"  # 旧版projection参数（PROJ4字符串）
    target_area = create_area_def(
        area_id="wgs84_60n60s",
        projection=wgs84_proj,  # 旧版参数：projection（PROJ4字符串）
        area_extent=area_extent,  # 保持您原有的区域范围
        resolution=resolution,
        description=f"WGS84 Lat/Lon, 60N-60S, {resolution}deg resolution"
    )
    return target_area


def _extract_time_from_filename(filename):
    """从MSG文件名中提取时间戳（格式：20251126062741 → 2025-11-26_06-27-41）"""
    try:
        # MSG文件名格式示例：MSG2-SEVI-MSG15-0100-NA-20251126062741.272000000Z-NA.nat
        time_part = filename.split("-")[5].split(".")[0]  # 提取"20251126062741"
        formatted_time = f"{time_part[:4]}-{time_part[4:6]}-{time_part[6:8]}_{time_part[8:10]}-{time_part[10:12]}-{time_part[12:14]}"
        return formatted_time
    except IndexError:
        print("警告：无法从文件名提取时间戳，使用默认时间格式")
        return "unknown_time"


def _process_msg_seviri_to_numpy(nat_file_path, resolution=0.05, resampler="nearest", channels=("WV_062",), lock=None):
    """
    核心处理函数：加载MSG数据→转换WGS84→返回numpy数组
    """
    try:
        if lock is not None:
            lock.acquire()

        # ========== 主要改动开始 ==========
        # 使用上下文管理器确保Scene资源被释放（替换原来的Scene创建）
        with satpy_scene_context(filenames=[nat_file_path], reader="seviri_l1b_native") as scn:
            scn.load(channels)

            ld = scn[channels[0]].attrs['orbital_parameters']['projection_longitude']
            area_extent = (ld - 60, -60, ld + 60, 60.0)
            target_area = _define_wgs84_area(resolution=resolution, area_extent=area_extent)

            scn_wgs84 = scn.resample(target_area, resampler=resampler)

            time_str = _extract_time_from_filename(os.path.basename(nat_file_path))

            result = {
                'data': {},
                'metadata': {},
                'coordinates': {},
                'global_attrs': {
                    'source_file': os.path.basename(nat_file_path),
                    'processing_time': time_str,
                    'resolution_degrees': resolution,
                    'resampling_method': resampler,
                    'area_extent': area_extent
                }
            }

            # 提取每个通道的数据
            for ch in channels:
                try:
                    data_array = scn_wgs84[ch].values
                    result['data'][ch] = data_array
                    result['metadata'][ch] = dict(scn_wgs84[ch].attrs)
                except Exception as e:
                    print(f"提取{ch}通道数据失败：{str(e)}")

            # 提取坐标信息
            if channels:
                first_ch = channels[0]
                try:
                    area = scn_wgs84[first_ch].attrs['area']
                    lons, lats = area.get_lonlats()
                    result['coordinates']['longitude'] = lons
                    result['coordinates']['latitude'] = lats
                    result['coordinates']['shape'] = lons.shape
                except Exception as e:
                    print(f"提取坐标信息失败：{str(e)}")

            return result
        # ========== 主要改动结束 ==========

    except Exception as e:
        print(f"处理失败：{str(e)}")
        return None
    finally:
        if lock is not None:
            lock.release()


def getNPfromNAT(file_path, save_file=False, lock=None):
    all_target_channels = ["WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"]
    # all_target_channels = ["VIS006", "VIS008"]
    result = _process_msg_seviri_to_numpy(nat_file_path=file_path, resolution=0.05, resampler="nearest",
                                          channels=all_target_channels, lock=lock)
    np_data = None
    coord = None
    if result is not None:
        for idx, channel in enumerate(all_target_channels):
            if channel in result['data']:
                chann_data = result['data'][channel]
                if np_data is None:
                    np_data = np.zeros((len(all_target_channels),) + chann_data.shape, dtype=chann_data.dtype)
                    area_extent = result['global_attrs']['area_extent']
                    coord = Coordinate(left=area_extent[0], bottom=area_extent[1], right=area_extent[2],
                                       top=area_extent[3], x_res=0.05, y_res=0.05)
                np_data[idx] = chann_data
            else:
                raise Exception(f"文件{file_path}，通道 {channel} 数据未找到")
    else:
        print("\n处理失败！")
    if save_file:
        np2tif(np_data, save_path='np2tif_dir', coord=coord, out_name='MetSat',
               dtype='float32')
    return np_data


def get_seviri_file_path(data_path, data_date):
    e_date = data_date + timedelta(minutes=14, seconds=59)
    parent_dir = rf'{data_path}/{data_date.strftime("%Y/%m/%d/")}'
    start_date_str = data_date.strftime('%Y%m%d%H%M%S')
    end_date_str = e_date.strftime('%Y%m%d%H%M%S')
    for file in os.listdir(parent_dir):
        if file.endswith('.nat') and start_date_str <= file.split('-')[5].split('.')[0] <= end_date_str:
            return os.path.join(parent_dir, file)
    return None


if __name__ == '__main__':
    np_data = getNPfromNAT("MSG4-SEVI-MSG15-0100-NA-20221230031243.610000000Z-NA.nat", save_file=False)
    print()
