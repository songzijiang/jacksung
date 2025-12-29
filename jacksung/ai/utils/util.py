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
import random


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


def get_stat_dict(metrics, extra_info=None):
    if extra_info is None:
        extra_info = dict()
    stat_dict = {'epochs': 0, 'loss': [], 'metrics': {}}
    for idx, metrics in enumerate(metrics):
        name, default_value, op = metrics
        stat_dict['metrics'][name] = {'value': [], 'best': {'value': default_value, 'epoch': 0, 'op': op}}
    for key, value in extra_info.items():
        stat_dict[key] = value
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


def draw_lines(stat_dict_path):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    print('[TemporaryTag]Producing the LinePicture of Log...', end='[TemporaryTag]\n')
    yaml_args = yaml.load(open(stat_dict_path), Loader=yaml.FullLoader)
    # 创建图表
    m_len = len(yaml_args['metrics'])
    sub_loss_count = 0
    for i in range(0, 5):
        if f'loss{i}es' in yaml_args:
            sub_loss_count += 1
        else:
            break
    plt.figure(figsize=(8 * (m_len + sub_loss_count), 6))  # 设置图表的大小
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
        # 把X坐标的纯统计范围缩放到和其他图表一致(epoch)
        x = np.array(range(1, len(y) + 1)) / scale
        plt.plot(x, y)
        plt.title(f'Loss{i}')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # 添加图例
    # plt.legend()
    plt.savefig(os.path.join(os.path.dirname(stat_dict_path), 'Metrics.jpg'))


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


def data_augmentation(images):
    # 旋转
    rotate = random.random()
    if 0 <= rotate < 0.25:
        images = [torch.rot90(image, 1, [2, 3]) for image in images]
    elif 0.25 <= rotate < 0.5:
        images = [torch.rot90(image, 2, [2, 3]) for image in images]
    elif 0.5 <= rotate < 0.75:
        images = [torch.rot90(image, 3, [2, 3]) for image in images]
    # 水平翻折
    if random.random() > 0.5:
        images = [torch.flip(image, [2]) for image in images]
    # 垂直翻折
    if random.random() > 0.5:
        images = [torch.flip(image, [3]) for image in images]
    return images


def clipSatelliteNP(np_data, ld, area=((100, 140, 10), (20, 60, 10))):
    lon_d = int((ld - (area[0][0] + area[0][1]) / 2) * 20)
    lat_d = int(((area[1][0] + area[1][1]) / 2) * 20)
    np_data = np_data[:, 800 - lat_d:1600 - lat_d, 800 - lon_d:1600 - lon_d]
    return np_data
