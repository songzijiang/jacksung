import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from jacksung.utils.image import crop_png, zoom_image, zoomAndDock
import rasterio
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import yaml
import argparse
from datetime import datetime, timedelta
from matplotlib.ticker import MaxNLocator


def load_model(model, state_dict, strict=True):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        name = name[name.index('.') + 1:]
        if name in own_state.keys():
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
                # own_state[name].requires_grad = False
            except Exception as e:
                err_log = f'While copying the parameter named {name}, ' \
                          f'whose dimensions in the model are {own_state[name].size()} and ' \
                          f'whose dimensions in the checkpoint are {param.size()}.'
                if not strict:
                    print(err_log)
                else:
                    raise Exception(err_log)
        elif strict:
            raise KeyError(f'unexpected key {name} in {own_state.keys()}')
        else:
            print(f'{name} not loaded by model')


def get_stat_dict(metrics):
    stat_dict = {
        'epochs': 0, 'loss0es': [], 'loss1es': [], 'loss2es': [], 'metrics': {}}
    for idx, metrics in enumerate(metrics):
        name, default_value, op = metrics
        stat_dict['metrics'][name] = {'value': [], 'best': {'value': default_value, 'epoch': 0, 'op': op}}
    return stat_dict


def data_to_device(datas, device, fp=32):
    outs = []
    for data in datas:
        if fp == 16:
            data = data.type(torch.HalfTensor)
        if fp == 64:
            data = data.type(torch.DoubleTensor)
        if fp == 32:
            data = data.type(torch.FloatTensor)
        data = data.to(device)
        outs.append(data)
    return outs


def draw_lines(yaml_path):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print('[TemporaryTag]Producing the LinePicture of Log...', end='[TemporaryTag]\n')
    yaml_args = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    # 创建图表
    m_len = len(yaml_args['metrics'])
    sub_loss_count = 0
    for i in range(0, 5):
        if f'loss{i}es' in yaml_args:
            sub_loss_count += 1
        else:
            break
    plt.figure(figsize=(10 * (m_len + sub_loss_count), 6))  # 设置图表的大小
    x = np.array(range(1, yaml_args['epochs'] + 1))
    for idx, d in enumerate(yaml_args['metrics'].items()):
        m_name, m_value = d
        y = np.array(m_value['value'])
        # 生成数据
        plt.subplot(1, m_len + sub_loss_count, idx + 1)
        plt.plot(x, y)
        plt.title(m_name)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(0, sub_loss_count):
        y = np.array(yaml_args[f'loss{i}es'])
        plt.subplot(1, m_len + sub_loss_count, m_len + i + 1)
        scale = len(y) / yaml_args['epochs']
        x = np.array(range(1, len(y) + 1)) / scale
        plt.plot(x, y)
        plt.title(f'Loss{i}')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # 添加图例
    # plt.legend()
    plt.savefig(os.path.join(os.path.dirname(yaml_path), 'Metrics.jpg'))


def make_best_metric(stat_dict, metrics, epoch, save_model_param, server_log_param):
    save_model_flag = False
    experiment_model_path, model, optimizer, scheduler = save_model_param
    log, epochs, cloudLogName = server_log_param

    for name, m_value in metrics:
        stat_dict['metrics'][name]['value'].append(m_value)
        inf = float('inf')
        if eval(str(m_value) + stat_dict['metrics'][name]['best']['op'] + str(
                stat_dict['metrics'][name]['best']['value'])):
            stat_dict['metrics'][name]['best']['value'] = m_value
            stat_dict['metrics'][name]['best']['epoch'] = epoch
            log.send_log('{}:{} epoch:{}/{}'.format(name, m_value, epoch, epochs), cloudLogName)
            save_model_flag = True

    if save_model_flag:
        # sava best model
        save_model(os.path.join(experiment_model_path, 'model_{}.pt'.format(epoch)), epoch,
                   model, optimizer, scheduler, stat_dict)
    # '[Validation] nRMSE/RMSE: {:.4f}/{:.4f} (Best: {:.4f}/{:.4f}, Epoch: {}/{})\n'
    test_log = f'[Val epoch:{epoch}] ' \
               + ' '.join([str(m[0]) + ':' + str(round(m[1], 4)) + '('
                           + str(round(stat_dict['metrics'][m[0]]['best']['value'], 4)) + ')' for m in metrics]) \
               + ' (Best Epoch: ' \
               + '/'.join([str(stat_dict['metrics'][m[0]]['best']['epoch']) for m in metrics]) \
               + ')'
    save_model(os.path.join(experiment_model_path, 'model_latest.pt'), epoch, model, optimizer, scheduler, stat_dict)
    return test_log


def save_model(_path, _epoch, _model, _optimizer, _scheduler, _stat_dict):
    # torch.save(model.state_dict(), saved_model_path)
    torch.save({
        'epoch': _epoch,
        'model_state_dict': _model.state_dict(),
        'optimizer_state_dict': _optimizer.state_dict(),
        'scheduler_state_dict': _scheduler.state_dict(),
        'stat_dict': _stat_dict
    }, _path)


def parse_config(config=None, set_gpu=True):
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config', type=str, default=None, help='pre-config file for training')
    parser.add_argument('--prec_data_path', type=str, default=None, help='dataset path')
    args = parser.parse_args()
    if args.config:
        opt = vars(args)
        yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(yaml_args)
    else:
        opt = vars(args)
        yaml_args = yaml.load(open(config), Loader=yaml.FullLoader)
        opt.update(yaml_args)

    if set_gpu:
        # set visible gpu
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.gpu_ids])

        # select active gpu devices
        if args.gpu_ids is not None and torch.cuda.is_available():
            print('use cuda & cudnn for acceleration!')
            print('the gpu id is: {}'.format(args.gpu_ids))
            device = torch.device('cuda')
            # device = torch.device('cuda:' + str(args.gpu_ids[0]))
        else:
            print('use cpu for training!')
            device = torch.device('cpu')
        return device, args
    else:
        return args


def _get_color_normalization(data, colors):
    max_value = colors[-1][0]
    min_value = colors[0][0]
    data[data < min_value] = min_value
    data[data > max_value] = max_value
    data = (data - min_value) / (max_value - min_value)
    new_colors = []
    for color in colors:
        new_colors.append([(color[0] - min_value) / (max_value - min_value), color[1]])
    return data, new_colors


def make_fig(file_name, root_path, out_folder=None, tz='UTC',
             colors=((0, '#1E90FF'), (0.1, '#1874CD'), (0.2, '#3A5FCD'), (0.3, '#0000CD'), (1, '#9400D3')),
             area=((100, 140, 10), (20, 60, 10)), font_size=20, corp=(0, 0, None, None),
             zoom_rectangle=(310 * 5, 300 * 5, 50 * 5, 40 * 5), docker=(300, 730), dpi=500, filter=0.3, exposure=None):
    # corp = [92, 31, 542, 456]
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    extents = [100, 140, 20, 60]
    proj = ccrs.PlateCarree()
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extents, crs=proj)
    # 读取TIFF数据
    elevation = None
    if type(file_name) == list:
        for each_file in file_name:
            file_path = os.path.join(root_path, each_file)
            with rasterio.open(file_path) as dataset:
                el_rd = dataset.read(1)
                elevation[elevation < filter] = np.nan
                if elevation is None:
                    elevation = el_rd
                else:
                    elevation += el_rd
    else:
        file_path = os.path.join(root_path, file_name)
        with rasterio.open(file_path) as dataset:
            elevation = dataset.read(1)
        elevation[elevation <= filter] = np.nan
    elevation, colors = _get_color_normalization(elevation, colors)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    # 添加各种特征
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='none',
                                        linewidth=0.4)
    ax.add_feature(land)
    ax.imshow(elevation, origin='upper', extent=extents, transform=proj, cmap=cmap)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)
    # ax.add_feature(cfeature.BORDERS)
    # 添加网格线
    # ax.gridlines(linestyle='--')
    # 设置大刻度和小刻度
    tick_proj = ccrs.PlateCarree()
    ax.set_xticks(np.arange(area[0][0], area[0][1] + 1, area[0][2]), crs=tick_proj)
    # ax.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=tick_proj)
    ax.set_yticks(np.arange(area[1][0], area[1][1] + 1, area[1][2]), crs=tick_proj)
    # ax.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=tick_proj)
    # 利用Formatter格式化刻度标签
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    if out_folder is None:
        if type(file_name) == list:
            file_name = file_name[0]
            file_dir = 'concate'
        else:
            file_dir = 'outs'
    else:
        file_dir = out_folder
    os.makedirs(os.path.join(root_path, file_dir), exist_ok=True)
    file_name = file_name.replace('.tif', '.png')
    file_title = datetime.strptime(file_name.split('-')[0].replace('target_', ''), '%Y%m%d_%H%M%S')
    file_name = file_title.strftime('%Y%m%d_%H%M%S.png')
    exposure = exposure if exposure else (60 if file_dir == 'concate' else 15)
    if tz == 'BJT':
        exposure_end = (file_title + timedelta(hours=8) + timedelta(minutes=exposure)).strftime('%H:%M')
        file_title = (file_title + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')
    else:
        exposure_end = (file_title + timedelta(minutes=exposure)).strftime('%H:%M')
        file_title = file_title.strftime('%Y-%m-%d %H:%M')
    ax.set_title(file_title + f'-' + exposure_end + f' ({tz})', fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    # plt.title(fontsize=font_size)
    file_save_path = os.path.join(root_path, file_dir, file_name)
    plt.savefig(file_save_path)
    if zoom_rectangle is not None:
        read_png = cv2.imread(file_save_path)
        read_png = zoomAndDock(read_png, zoom_rectangle, docker, scale_factor=5, border=14)
        cv2.imwrite(file_save_path, read_png)
    crop_png(file_save_path, left=corp[0], top=corp[1], right=corp[2], bottom=corp[3])
    return file_save_path
    # plt.show()
