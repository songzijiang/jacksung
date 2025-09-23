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

    def calc_AUROC(self, preds, target):
        AUROC = 0
        for i in range(len(preds)):
            AUROC += self.AUROC(preds[i].flatten(), target[i].flatten())
        self.AUROC.reset()
        return AUROC / len(preds)

    def calc_psnr(self, preds, target):
        psnr = 0
        for i in range(len(preds)):
            psnr += self.psnr(rearrange(preds[i], '(b c) h w->b c h w', b=1),
                              rearrange(target[i], '(b c) h w->b c h w', b=1))
        self.psnr.reset()
        return psnr / len(preds)

    def calc_ssim(self, preds, target):
        ssim = 0
        for i in range(len(preds)):
            ssim += self.ssim(rearrange(preds[i], '(b c) h w->b c h w', b=1),
                              rearrange(target[i], '(b c) h w->b c h w', b=1))
        self.ssim.reset()
        return ssim / len(preds)

    def calc_rmse(self, preds, target):
        rmse = 0
        for i in range(len(preds)):
            rmse += compute_rmse(preds[i], target[i])
        return rmse / len(preds)

    def calc_rr(self, preds, target):
        rr = 0
        for i in range(len(preds)):
            rr += self.rr(preds[i].flatten(), target[i].flatten())
        self.rr.reset()
        return rr / len(preds)

    def calc_p(self, preds, target, exclude_zero=False):
        p = 0
        count = 0
        for i in range(len(preds)):
            pred, target = preds[i].flatten(), target[i].flatten()
            if exclude_zero:
                mask = target != 0
                pred = pred[mask]
                target = target[mask]
            if pred.var() == 0 or target.var() == 0:
                continue
            count += 1
            p += self.p(pred, target)
        self.p.reset()
        return p / count

    def print_metrics(self, preds, target, print_log=True):
        rr = float(self.calc_rr(preds, target))
        p = float(self.calc_p(preds, target))
        rmse = float(self.calc_rmse(preds, target))
        ssim = float(self.calc_ssim(preds, target))
        psnr = float(self.calc_psnr(preds, target))
        if print_log:
            print(rf'p: {p} rr: {rr} rmse: {rmse} ssim: {ssim} psnr: {psnr}')
        return {'p': p, 'rr': rr, 'rmse': rmse, 'ssim': ssim, 'psnr': psnr}

    def calculate_rain_metrics(self, preds, target, threshold=0.1):
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
        target_flat = target.flatten(start_dim=1)

        # 2. 二值化（1=有雨，0=无雨）
        preds_binary = (preds_flat >= threshold).float()
        target_binary = (target_flat >= threshold).float()

        # 3. 计算混淆矩阵元素（按样本维度求和）
        TP = torch.sum(preds_binary * target_binary, dim=1)  # 每个样本的TP总和
        FP = torch.sum(preds_binary * (1 - target_binary), dim=1)
        TN = torch.sum((1 - preds_binary) * (1 - target_binary), dim=1)
        FN = torch.sum((1 - preds_binary) * target_binary, dim=1)

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

        return {
            'POD': POD,
            'FAR': FAR,
            'ACC': ACC,
            'CSI': CSI
        }


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
                            [0, 1, 1]
                            ]])
    # m = Metrics()
    # print(m.calc_rr(preds, target))
    m = Metrics()
    AUROC = m.calc_AUROC(preds, target)
    print(AUROC)
    print(m.calculate_rain_metrics(preds, target))