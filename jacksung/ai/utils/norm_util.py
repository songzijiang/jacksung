import os.path

import torch
import numpy as np
from einops import rearrange


class PredNormalization:
    def __init__(self, data_path):
        self.mean = np.load(os.path.join(data_path, 'mean_level.npy')).astype(np.float32)[2:]
        self.std = np.load(os.path.join(data_path, 'std_level.npy')).astype(np.float32)[2:]
        self.mean = torch.from_numpy(self.mean)
        self.std = torch.from_numpy(self.std)

    def norm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = (data - self.mean) / self.std
        return rearrange(data, 'b h w c->b c h w')

    def denorm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = data * self.std + self.mean
        return rearrange(data, 'b h w c->b c h w')


class PrecNormalization:
    def __init__(self, data_path):
        self.mean_qpe = torch.from_numpy(
            np.load(os.path.join(data_path, 'mean_level_qpe.npy')).astype(np.float32))
        self.std_qpe = torch.from_numpy(
            np.load(os.path.join(data_path, 'std_level_qpe.npy')).astype(np.float32))
        self.mean_fy = torch.from_numpy(
            np.load(os.path.join(data_path, 'mean_level_fy.npy')).astype(np.float32).mean(axis=0)[2:])
        self.std_fy = torch.from_numpy(
            np.load(os.path.join(data_path, 'std_level_fy.npy')).astype(np.float32).mean(axis=0)[2:])

    def norm(self, data, norm_type='fy'):
        if norm_type == 'fy':
            data = rearrange(data, 'b t c h w->b h w t c')
            data = (data - self.mean_fy) / self.std_fy
            return rearrange(data, 'b h w t c->b t c h w')
        elif norm_type == 'qpe':
            data = rearrange(data, 'b c h w->b h w c')
            data = (data - self.mean_qpe) / self.std_qpe
            return rearrange(data, 'b h w c->b c h w')

    def denorm(self, data, norm_type='fy'):
        if norm_type == 'fy':
            data = rearrange(data, 'b t c h w->b h w t c')
            data = data * self.std_fy + self.mean_fy
            return rearrange(data, 'b h w t c->b t c h w')
        elif norm_type == 'qpe':
            data = rearrange(data, 'b c h w->b h w c')
            data = data * self.std_qpe + self.mean_qpe
            return rearrange(data, 'b h w c->b c h w')


class PremNormalization:

    def __init__(self, prec_data_path):
        self.mean = torch.from_numpy(
            np.load(os.path.join(prec_data_path, 'imerg_mean.npy')).astype(np.float32))
        self.std = torch.from_numpy(
            np.load(os.path.join(prec_data_path, 'imerg_std.npy')).astype(np.float32))
        # self.mean = torch.Tensor(
        #     [3.6614e+03, 3.3748e+03, 3.3829e+03, 2.8368e+03, 2.5664e+03, 2.4914e+03, 2.4259e+03, 0.5723, 0, 0])
        # self.std = torch.Tensor([164.7376, 265.8857, 252.9820, 509.8994, 532.4901, 518.8191, 414.1427, 0.6308, 1, 1])
        # print(self.mean, self.std)

    def norm(self, data, fy_norm=True):
        data = rearrange(data, 'b c h w->b h w c')
        if fy_norm:
            data = (data - self.mean[:7]) / self.std[:7]
        else:
            # data[:, 0][data[:, 0] == 0] = torch.inf
            # data[:, 0] = 1 / torch.sqrt(data[:, 0].clone())
            data = (data - self.mean[7:]) / self.std[7:]
        return rearrange(data, 'b h w c->b c h w')

    def denorm(self, data, fy_norm=True):
        data = rearrange(data, 'b c h w->b h w c')
        if fy_norm:
            data = data * self.std[:7] + self.mean[:7]
        else:
            data = data * self.std[7:] + self.mean[7:]
            # data[:, 0][data[:, 0] == 0] = torch.inf
            # data[:, 0] = 1 / (data[:, 0].clone() ** 2)
        return rearrange(data, 'b h w c->b c h w')


class Normalization:
    def __init__(self, mean_std_npy, idx=None):
        mean_std = torch.from_numpy(mean_std_npy.astype(np.float32))
        if idx:
            self.mean = mean_std[0, idx[0]:idx[1]]
            self.std = mean_std[1, idx[0]:idx[1]]
        else:
            self.mean = mean_std[0]
            self.std = mean_std[1]

    def norm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = (data - self.mean) / self.std
        return rearrange(data, 'b h w c->b c h w')

    def denorm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = data * self.std + self.mean
        return rearrange(data, 'b h w c->b c h w')
