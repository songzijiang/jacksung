import torch.nn as nn
from jacksung.ai.GeoNet.m_blockV2 import FEB, Tail, Head, DownBlock, UpBlock, CubeEmbeding, CubeUnEmbeding, Norm, ACT
import torch
import jacksung.utils.fastnumpy as fnp


class GeoNet(nn.Module):
    def __init__(self, window_sizes, n_lgab, c_in, c_lgan, r_expand=4, down_sample=2, num_heads=8, downstage=2):
        super(GeoNet, self).__init__()
        self.window_sizes = window_sizes
        self.n_lgab = n_lgab
        self.c_in = c_in
        self.c_lgan = c_lgan
        self.r_expand = r_expand
        self.down_sample = down_sample
        # define head module
        self.head = Head(self.c_lgan, self.c_in, self.down_sample)
        # define body module
        self.body = nn.ModuleList()
        self.downstage = downstage
        for i in range(self.n_lgab):
            if i % self.downstage in [1]:
                if 0 <= i < self.n_lgab / 2 - 1:
                    self.body.append(DownBlock(self.c_lgan, 2))
                    self.c_lgan = self.c_lgan * 2
                elif i > self.n_lgab / 2 + 1:
                    self.body.append(UpBlock(self.c_lgan // 2, 2))
                    self.c_lgan = self.c_lgan // 2
            self.body.append(
                FEB(self.c_lgan, self.r_expand, self.window_sizes[i % len(self.window_sizes)], num_heads=num_heads))
        self.tail = Tail(self.c_lgan, 3, down_sample)

    def forward(self, x, roll=0):
        # head

        x = torch.roll(x, shifts=roll, dims=-1)
        x = self.head(x)
        x_res = list()
        x_res.append(x)
        idx = 0
        for stage in self.body:
            if str(stage.__class__) == "<class 'models.GeoNet.m_blockV2.FEB'>":
                # 7,9
                if idx > self.n_lgab / 2 + 1 and idx % self.downstage in [1]:
                    x += x_res.pop()
                # 0,2
                x = stage(x)
                if idx < self.n_lgab / 2 - 1 and idx % self.downstage in [0]:
                    x_res.append(x)
                idx += 1
            else:
                x = stage(x)
        # tail
        x += x_res.pop()
        x = self.tail(x)
        if roll > 0:
            x = torch.roll(x, shifts=-roll, dims=-1)
        return x

    def init_model(self):
        print('Initializing the model!')
        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

    def load(self, state_dict, strict=True):
        own_state = self.state_dict()
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


if __name__ == '__main__':
    pass
