import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from pytorch_msssim import ssim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import R2Score, PearsonCorrCoef, AUROC
from torchmetrics.regression import MeanSquaredError
import importlib
from einops import rearrange
import cv2

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def uncertainty_bootstrap(obs, sim, metric_func, n_boot=1000, threshold=0.1):
    """
    标准Bootstrap（每次抽n个样本）
    Parameters
    ----------
    obs : array-like
        观测值
    sim : array-like
        模拟值
    metric_func : function
        指标计算函数（如calculate_metrics）
    n_boot : int
        bootstrap次数
    Returns
    -------
    dict
        各指标的 mean / std / 95% CI
    """
    obs = np.array(obs)
    sim = np.array(sim)
    n = len(obs)
    results = []
    for _ in range(n_boot):
        # 有放回抽样
        idx = np.random.randint(0, n, n)
        obs_sample = obs[idx]
        sim_sample = sim[idx]
        metrics = metric_func(obs_sample, sim_sample, threshold)
        results.append(metrics)
    # 转为dict of lists
    metric_names = results[0].keys()
    stats = {}
    for m in metric_names:
        values = np.array([r[m] for r in results])
        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)
        stats[m] = rf"{round((ci_low + ci_high) / 2, 3)}±{round((ci_high - ci_low) / 2, 3)}"
    return stats


def precipitation_metrics(obs_data, sim_data, threshold=0.1, interval_min=-1, interval_max=1000, digits=4):
    """
    计算降水评估指标：CSI, FAR, POD, ACC, CC, R², RMSE
    Parameters:
    -----------
    obs_data : list or np.array
        观测值序列（station数据）
    sim_data : list or np.array
        模拟值序列（栅格数据，如HHR、cmorph等）
    threshold : float
        降水阈值，默认0.1mm，大于等于阈值视为有雨
    Returns:
    --------
    dict
        包含各指标计算结果的字典
    """
    if len(obs_data) != len(sim_data):
        raise ValueError(rf"观测值和模拟值长度不一致,{len(obs_data)} vs {len(sim_data)}")
    # 转换为numpy数组，方便计算
    obs = np.array(obs_data, dtype=float)
    sim = np.array(sim_data, dtype=float)
    obs[obs < 0] = np.nan
    sim[sim < 0] = np.nan
    obs[obs > 1000] = np.nan
    sim[sim > 1000] = np.nan
    # 分区间过滤
    obs[obs <= interval_min] = np.nan
    obs[obs > interval_max] = np.nan
    # 过滤掉NaN和None值（无效数据）
    valid_mask = ~(np.isnan(obs) | np.isnan(sim) | np.isinf(obs) | np.isinf(sim))
    obs_valid = obs[valid_mask]
    sim_valid = sim[valid_mask]
    if len(obs_valid) < 10:
        return {
            'CSI': np.nan, 'FAR': np.nan, 'POD': np.nan,
            'ACC': np.nan, 'CC': np.nan, 'R2': np.nan, 'RMSE': np.nan
        }
    # 区分有雨/无雨（基于阈值）
    obs_rain = obs_valid > threshold
    sim_rain = sim_valid > threshold
    # 计算h, m, f, r
    h = np.sum(obs_rain & sim_rain)  # 击中
    m = np.sum(obs_rain & ~sim_rain)  # 漏报
    f = np.sum(~obs_rain & sim_rain)  # 误报
    r = np.sum(~obs_rain & ~sim_rain)  # 正确否定
    # 计算传统指标（避免分母为0）
    POD = h / (h + m) if (h + m) != 0 else np.nan
    FAR = f / (h + f) if (h + f) != 0 else np.nan
    CSI = h / (h + m + f) if (h + m + f) != 0 else np.nan
    ACC = (h + r) / (h + m + f + r) if (h + m + f + r) != 0 else np.nan
    # 计算皮尔逊相关系数(CC)和决定系数(R²)
    CC, _ = pearsonr(obs_valid, sim_valid)
    RMSE = np.sqrt(np.mean((sim_valid - obs_valid) ** 2))
    R2 = r2_score(obs_valid, sim_valid)
    return {
        'CSI': round(CSI, digits),
        'FAR': round(FAR, digits),
        'POD': round(POD, digits),
        'ACC': round(ACC, digits),
        'CC': round(CC, digits),
        'R2': round(R2, digits),
        'RMSE': round(RMSE, digits)
    }


def compute_rmse(da_fc, da_true):
    error = da_fc - da_true
    error = error ** 2
    number = torch.sqrt(error.mean((-2, -1)))
    return number.mean()


class Metrics:
    def __init__(self):
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.rr = R2Score()
        self.p = PearsonCorrCoef()
        self.AUROC = AUROC("binary")

    def mask_nan(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()
        # 生成有效掩码：真实值和预测值均非 NaN 的位置为 True
        valid_mask = ~(torch.isnan(pred) | torch.isnan(target))

        # 过滤无效元素
        pred = pred[valid_mask]
        target = target[valid_mask]

        return pred, target

    def calc_AUROC(self, preds, targets):
        AUROC = 0
        for i in range(len(preds)):
            pred, target = self.mask_nan(preds[i], targets[i])
            AUROC += self.AUROC(pred, target)
        self.AUROC.reset()
        return AUROC / len(preds)

    def calc_psnr(self, preds, targets):
        psnr = 0
        for i in range(len(preds)):
            psnr += self.psnr(rearrange(preds[i], '(b c) h w->b c h w', b=1),
                              rearrange(targets[i], '(b c) h w->b c h w', b=1))
        self.psnr.reset()
        return psnr / len(preds)

    def calc_ssim(self, preds, targets):
        ssim = 0
        for i in range(len(preds)):
            ssim += self.ssim(rearrange(preds[i], '(b c) h w->b c h w', b=1),
                              rearrange(targets[i], '(b c) h w->b c h w', b=1))
        self.ssim.reset()
        return ssim / len(preds)

    def calc_rmse(self, preds, targets):
        rmse = 0
        for i in range(len(preds)):
            rmse += compute_rmse(preds[i], targets[i])
        return rmse / len(preds)

    def calc_rr(self, preds, targets):
        rr = 0
        for i in range(len(preds)):
            pred, tar = self.mask_nan(preds[i], targets[i])
            rr += self.rr(pred, tar)
        self.rr.reset()
        return rr / len(preds)

    def calc_p(self, preds, targets, exclude_zero=False):
        p = 0
        count = 0
        for i in range(len(preds)):
            pred, target = self.mask_nan(preds[i], targets[i])
            if exclude_zero:
                mask = target != 0
                pred = pred[mask]
                target = target[mask]
            if pred.var() == 0 or target.var() == 0:
                continue
            count += 1
            p += self.p(pred, target)
        self.p.reset()
        if count == 0:
            return torch.nan
        return p / count

    def print_metrics(self, preds, targets, print_log=True):
        rr = float(self.calc_rr(preds, targets))
        p = float(self.calc_p(preds, targets))
        rmse = float(self.calc_rmse(preds, targets))
        ssim = float(self.calc_ssim(preds, targets))
        psnr = float(self.calc_psnr(preds, targets))
        if print_log:
            print(rf'p: {p} rr: {rr} rmse: {rmse} ssim: {ssim} psnr: {psnr}')
        return {'p': p, 'rr': rr, 'rmse': rmse, 'ssim': ssim, 'psnr': psnr}

    def calculate_rain_metrics(self, preds, targets, threshold=0.1):
        """
        使用flatten()批量计算多张降雨图的POD、FAR、ACC、CSI指标

        参数:
            preds: 预测的降雨tensor，形状为[样本数, 高度, 宽度]
            target: 观测的降雨tensor，形状与preds相同
            threshold: 降雨事件的阈值，默认0.1mm

        返回:
            metrics: 包含每个样本指标的字典
        """
        # 1. 将每个样本的空间维度展平（[样本数, 高度, 宽度] → [样本数, 像素总数]）
        preds_flat = preds.flatten(start_dim=1)  # 从第1维开始展平（保留样本维度）
        targets_flat = targets.flatten(start_dim=1)
        # 2. 二值化（1=有雨，0=无雨）
        preds_binary = (preds_flat > threshold).float()
        targets_binary = (targets_flat > threshold).float()

        # 3. 计算混淆矩阵元素（按样本维度求和）
        TP = torch.sum(preds_binary * targets_binary, dim=1)  # 每个样本的TP总和
        FP = torch.sum(preds_binary * (1 - targets_binary), dim=1)
        TN = torch.sum((1 - preds_binary) * (1 - targets_binary), dim=1)
        FN = torch.sum((1 - preds_binary) * targets_binary, dim=1)

        # 4. 计算指标（处理分母为0的情况）
        POD = TP / (TP + FN)
        FAR = FP / (TP + FP)
        ACC = (TP + TN) / (TP + FP + TN + FN)
        CSI = TP / (TP + FP + FN)

        # 5. 标记无效值为NaN
        POD = torch.where((TP + FN) == 0, torch.nan, POD)
        FAR = torch.where((TP + FP) == 0, torch.nan, FAR)
        ACC = torch.where((TP + FP + TN + FN) == 0, torch.nan, ACC)
        CSI = torch.where((TP + FP + FN) == 0, torch.nan, CSI)

        return POD, FAR, ACC, CSI


def img2tensor(img):
    if type(img) == str:
        img = cv2.imread(img, -1)
    img = torch.from_numpy(img)
    img = rearrange(img, ' (b c h) w->b c h w', b=1, c=1)
    return img


if __name__ == '__main__':
    preds = torch.Tensor([[[0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]
                           ],
                          [[0, 1, 1],
                           [0, 1, 1],
                           [0, 1, 1]
                           ]])
    # target = torch.rand(2, 1, 3, 3)
    target = torch.Tensor([[[0, 1, 1],
                            [0, 1, 1],
                            [0, 1, 1]
                            ],
                           [[0, 1, 1],
                            [0, 1, 1],
                            [0, 0, 1]
                            ]])
    target[1, 0, 2] = torch.nan
    # m = Metrics()
    # print(m.calc_rr(preds, target))
    m = Metrics()
    AUROC = m.calc_AUROC(preds, target)
    print(AUROC)
    print(m.calculate_rain_metrics(preds, target))
