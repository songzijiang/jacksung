import os.path
import random
from datetime import timedelta

import torch
from jacksung.utils.time import RemainTime, Stopwatch, cur_timestamp_str
from jacksung.ai.utils.norm_util import PredNormalization, PrecNormalization, PremNormalization
import numpy as np
from jacksung.utils.data_convert import np2tif
from jacksung.utils.cache import Cache
from jacksung.ai.utils.fy import prase_filename, getFY_coord_clip, getNPfromHDFClip
from einops import rearrange
from jacksung.ai.utils.util import parse_config, data_to_device
from jacksung.ai.GeoNet.m_network import GeoNet
from jacksung.ai.GeoNet.m_networkV2 import GeoNet as GeoNetV2
from jacksung.utils.exception import NoFileException, NanNPException
import torch.nn as nn


def _get_np_array(fy_npy):
    if type(fy_npy) is str:
        print(f'正在反演:{fy_npy}...')
        if not os.path.exists(fy_npy):
            raise NoFileException(fy_npy)
        n_data = np.load(fy_npy)
    elif type(fy_npy) is np.ndarray:
        n_data = fy_npy
    else:
        raise Exception('输入数据类型错误，仅支持文件路径或numpy数组')
    return n_data


class GeoAttX:
    def __init__(self, config=None, root_path=None, task_type=None, area=((100, 140, 10), (20, 60, 10))):
        self.root_path = None
        self.timestamp = None
        self.dir_name = None
        self.device, self.args = parse_config(config)
        self.task_type = task_type
        self.set_root_path(root_path)
        self.area = area

    def get_root_path(self):
        return self.root_path

    def get_dir_name(self):
        return self.dir_name

    def set_root_path(self, root_path=None, dir_name=None):
        self.timestamp = cur_timestamp_str()
        root_path = root_path if root_path else self.args.save_path
        dir_name = dir_name if dir_name else self.task_type + '-' + self.args.model + '-' + self.timestamp + '_' + str(
            random.randint(1000, 9999))
        self.dir_name = dir_name
        self.root_path = os.path.join(root_path, dir_name)

    def load_model(self, path, version=1):
        if version == 1:
            model = GeoNet(window_sizes=self.args.window_sizes, n_lgab=self.args.n_lgab, c_in=self.args.c_in,
                           c_lgan=self.args.c_lgan, r_expand=self.args.r_expand, down_sample=self.args.down_sample,
                           num_heads=self.args.num_heads, task=self.task_type if self.task_type else self.args.task,
                           downstage=self.args.downstage)
        else:
            model = GeoNetV2(window_sizes=self.args.window_sizes, n_lgab=self.args.n_lgab, c_in=self.args.c_in,
                             c_lgan=self.args.c_lgan, r_expand=self.args.r_expand, down_sample=self.args.down_sample,
                             num_heads=self.args.num_heads, downstage=self.args.downstage)

        ckpt = torch.load(path, map_location=torch.device(self.device))
        model.load(ckpt['model_state_dict'])
        model = model.to(self.device)
        model = model.eval()
        model.requires_grad_(False)
        return model


class GeoAttX_I(GeoAttX):
    def __init__(self, data_path, x1_path, x4_path, x12_path, root_path=None, config='config_predict.yml',
                 area=((100, 140, 10), (20, 60, 10)), cache_size=1):
        super().__init__(config=config, root_path=root_path, task_type='pred', area=area)
        self.f, self.n, self.ys = None, None, None
        self.data_path = data_path
        self.x1 = self.load_model(x1_path)
        self.x4 = self.load_model(x4_path)
        self.x12 = self.load_model(x12_path)
        # self.x48 = self.load_model(x48_path)
        self.norm = PredNormalization(self.args.pred_data_path)
        self.norm.mean, self.norm.std = data_to_device([self.norm.mean, self.norm.std], self.device, self.args.fp)
        self.ld = None
        self.cache = Cache(cache_size)

    def save(self, file_name, ys):
        file_info = prase_filename(file_name)
        ld = int(file_info["position"])
        for idx, (k, y) in enumerate(ys.items()):
            # coord = getFY_coord_min(ld)
            coord = getFY_coord_clip(self.area)
            np2tif(y, save_path=self.root_path, out_name=f'{k.strftime("%Y%m%d_%H%M%S")}', coord=coord,
                   dtype=np.float32, print_log=False, dim_value=[{'value': [str(x) for x in list(range(9, 16))]}])
            td = k - file_info['start']
            mins = td.days * 24 * 60 + td.seconds // 60
            target_filename = self.get_filename(file_name, mins)
            p_path = self.get_path_by_filename(target_filename)
            if idx >= 1 and os.path.exists(p_path):
                coord = getFY_coord_clip(self.area)
                p_data = getNPfromHDFClip(self.ld, p_path, area=self.area)
                if p_data is not None:
                    p_data = p_data[2:, :, :]
                    np2tif(p_data, save_path=self.root_path, out_name=f'target_{k.strftime("%Y%m%d_%H%M%S")}',
                           print_log=False, coord=coord, dtype=np.float32,
                           dim_value=[{'value': [str(x) for x in list(range(9, 16))]}])
        print(f'data saved in {self.root_path}')
        with open(os.path.join(self.root_path, 'info.log'), 'w') as f:
            f.write(f'输入数据：{file_info["start"]} {file_info["position"]} {file_info["end"]}\n')
            for k, y in ys.items():
                f.write(f'预测：{k}\n')
        return self.root_path

    def get_filename(self, file_name, mins):
        file_info = prase_filename(file_name)
        new_file_name = file_name.replace(
            f'{file_info["start"].strftime("%Y%m%d%H%M%S")}_{file_info["end"].strftime("%Y%m%d%H%M%S")}',
            f'{(file_info["start"] + timedelta(minutes=mins)).strftime("%Y%m%d%H%M%S")}_'
            f'{(file_info["end"] + timedelta(minutes=mins)).strftime("%Y%m%d%H%M%S")}')
        return new_file_name

    def get_path_by_filename(self, file_name):
        file_info = prase_filename(file_name)
        return f'{self.data_path}{os.sep}downloaded_file{os.sep}{file_info["start"].year}{os.sep}{file_info["start"].month}{os.sep}{file_info["start"].day}{os.sep}{file_name}'
        # return f'{self.data_path}/{file_name}'

    def numpy2tensor(self, f_data):
        f_data = torch.from_numpy(f_data)
        f = data_to_device([f_data], self.device, self.args.fp)[0]
        f = rearrange(f, '(b c) h w -> b c h w', b=1)
        return f

    def get_exist_by_filename_and_mins(self, file_name, mins):
        f_path = self.get_filename(file_name, mins)
        f_path = self.get_path_by_filename(f_path)
        if not os.path.exists(f_path):
            raise NoFileException(f_path)
        f_data = getNPfromHDFClip(self.ld, f_path, area=self.area, cache=self.cache)
        if type(f_data) is not str:
            f_data = f_data[2:, :, :]
        else:
            raise NanNPException(f_path)
        # f_data = zoom(f_data.astype(np.float32), (1, 1 / 5, 1 / 5))
        return self.numpy2tensor(f_data)

    def mean_std2Tensor(self, in_data, h, w):
        in_data = rearrange(in_data, '(b h w) c->b c h w', h=1, w=1)
        in_data = in_data.expand(1, 7, h, w)
        return in_data

    def secdOrderStd(self, t_data, mean0, std0, mean, std):
        h, w = t_data.shape[2], t_data.shape[3]
        t_data = (t_data - self.mean_std2Tensor(mean0, h, w)) / self.mean_std2Tensor(std0, h, w)
        t_data = t_data * self.mean_std2Tensor(std, h, w) + self.mean_std2Tensor(mean, h, w)
        return t_data

    def predict(self, file_name, step=360, p_steps=(48, 12, 4, 1), print_log=True):
        try:
            file_info = prase_filename(file_name)
            self.ld = int(file_info["position"])
            step = step // 15
            # if step > 24:
            #     step = 24
            if print_log:
                print(f'当前时刻:{file_info["start"]}\n预测长度:{step * 15}分钟')
            task_progress = []
            p_steps = sorted(p_steps, reverse=True)
            while step > 0:
                for p_step in p_steps:
                    if step >= p_step:
                        task_progress.append(p_step)
                        step -= p_step
                        break
            task_progress.reverse()
            if print_log:
                print(f'正在预测:{file_info["start"] + timedelta(minutes=sum(task_progress) * 15)}...')
            n = self.get_exist_by_filename_and_mins(file_name, 0)
            now_date = file_info["start"]
            porcess_list = {now_date: n.detach().cpu().numpy()[0]}
            n = self.norm.norm(n)
            o_mean = torch.mean(n, dim=(2, 3))
            o_std = torch.std(n, dim=(2, 3))
            for step in task_progress:
                pre_date = now_date - timedelta(minutes=15 * step)
                now_date += timedelta(minutes=15 * step)
                if print_log:
                    print(f'正在预测 {now_date}...')
                if pre_date in porcess_list:
                    f = porcess_list[pre_date]
                    f = self.numpy2tensor(f)
                else:
                    f = self.get_exist_by_filename_and_mins(file_name,
                                                            -int((file_info["start"] - pre_date).seconds / 60))
                f = self.norm.norm(f)
                st = Stopwatch()
                y_ = eval(f'self.x{step}(f, n)')
                if print_log:
                    print(f'预测耗时: {st.reset()} 秒')
                # mean and std再标准化
                y_mean = torch.mean(y_, dim=(2, 3))
                y_std = torch.std(y_, dim=(2, 3))
                # 二次标准化
                y_ = self.secdOrderStd(y_, y_mean, y_std, o_mean, o_std)
                # y_ = torch.Tensor(np.zeros((1, 7, 2400, 2400), dtype=np.float32)).to(self.device)
                # for i in range(5):
                #     for j in range(5):
                #         n_ = n[:, :, 480 * i:480 * (i + 1), 480 * j:480 * (j + 1)]
                #         f_ = f[:, :, 480 * i:480 * (i + 1), 480 * j:480 * (j + 1)]
                #         y_[:, :, 480 * i:480 * (i + 1), 480 * j:480 * (j + 1)] = eval(f'self.x{step}(f_, n_)')
                del f
                n = y_
                porcess_list[now_date] = self.norm.denorm(y_).detach().cpu().numpy()[0]
        except (NoFileException, NanNPException) as e:
            os.makedirs(self.root_path, exist_ok=True)
            with open(os.path.join(self.root_path, 'err.log'), 'a') as f:
                filename = e.file_name.split(os.sep)[-1]
                file_info = prase_filename(filename)
                f.write(f'{e.__class__}: {file_info["start"]} {e.file_name}\n')
            return {}
        return porcess_list


class GeoAttX_P(GeoAttX):
    def __init__(self, model_path, root_path=None, config='predict_qpe.yml', area=((100, 140, 10), (20, 60, 10))):
        super().__init__(config=config, root_path=root_path, task_type='prec', area=area)
        self.model = self.load_model(model_path)

    def save(self, y, save_name, info_log=True, print_log=True):
        np2tif(y, save_path=self.root_path, out_name=save_name, coord=getFY_coord_clip(self.area), dtype=np.float32,
               print_log=False, dim_value=[{'value': ['qpe']}])
        if print_log:
            print(f'data saved in {self.root_path}')
        if info_log:
            with open(os.path.join(self.root_path, 'info.log'), 'w') as f:
                f.write(f'QPE 反演：{save_name}\n')
        return self.root_path

    def predict(self, fy_npy):
        try:
            n_data = _get_np_array(fy_npy)
            n_data = torch.from_numpy(n_data)
            norm = PrecNormalization(self.args.prec_data_path)
            norm.mean_fy, norm.mean_qpe, norm.std_fy, norm.std_qpe = \
                data_to_device([norm.mean_fy, norm.mean_qpe, norm.std_fy, norm.std_qpe], self.device, self.args.fp)
            n_data = data_to_device([n_data], self.device, self.args.fp)[0]
            n_data = rearrange(n_data, '(b t c) h w -> b t c h w', b=1, t=1)
            n = norm.norm(n_data, norm_type='fy')[:, 0, :, :, :]
            y_ = self.model(n, n)
            y = norm.denorm(y_, norm_type='qpe').detach().cpu().numpy()[0]
            return y
        except NoFileException as e:
            os.makedirs(self.root_path, exist_ok=True)
            with open(os.path.join(self.root_path, 'err.log'), 'a') as f:
                filename = e.file_name.split(os.sep)[-1]
                file_info = prase_filename(filename)
                f.write(f'Not exist {file_info["start"]} {e.file_name}\n')
            return None


class GeoAttX_M(GeoAttX):
    def __init__(self, model_path, root_path=None, config='predict_imerg.yml', area=((100, 140, 10), (20, 60, 10))):
        super().__init__(config=config, root_path=root_path, task_type='prem', area=area)
        self.model = self.load_model(model_path, version=2)

    def save(self, y, save_name, info_log=True, print_log=True):
        np2tif(y, save_path=self.root_path, out_name=save_name, coord=getFY_coord_clip(self.area), dtype=np.float32,
               print_log=False, dim_value=[{'value': ['imerg']}])
        if print_log:
            print(f'data saved in {self.root_path}')
        if info_log:
            with open(os.path.join(self.root_path, 'info.log'), 'w') as f:
                f.write(f'Imerg 反演：{save_name}\n')
        return self.root_path

    def predict(self, fy_npy, smooth=True):
        try:
            n_data = _get_np_array(fy_npy)
            n_data = torch.from_numpy(n_data)
            norm = PremNormalization(self.args.prec_data_path)
            norm.mean, norm.std = data_to_device([norm.mean, norm.std], self.device, self.args.fp)
            n_data = data_to_device([n_data], self.device, self.args.fp)[0]
            n_data = rearrange(n_data, '(b c) h w -> b c h w', b=1)
            n = norm.norm(n_data, fy_norm=True)[:, :, :, :]
            ps = nn.PixelShuffle(2)
            ups = nn.PixelUnshuffle(2)
            if smooth:
                smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            n = ups(n)
            n = rearrange(n, 'b (c dsize) h w -> (b dsize) c h w', dsize=4)
            y_ = self.model(n)
            y_ = rearrange(y_, '(b dsize) c h w -> b (c dsize) h w', dsize=4)
            y_ = ps(y_)
            y = norm.denorm(y_, fy_norm=False)[0]
            y[0][y[1] > y[2]] = 0
            y[0][y[0] < 0] = 0
            y = rearrange(y[0], '(b h) w -> b h w', b=1)
            _, H, W = y.shape
            if smooth:
                y[0, 1:H - 1, 1:W - 1] = smooth(y)[0, 1:H - 1, 1:W - 1]
            return y.detach().cpu().numpy()
        except NoFileException as e:
            os.makedirs(self.root_path, exist_ok=True)
            with open(os.path.join(self.root_path, 'err.log'), 'a') as f:
                filename = e.file_name.split(os.sep)[-1]
                file_info = prase_filename(filename)
                f.write(f'Not exist {file_info["start"]} {e.file_name}\n')
            return None
