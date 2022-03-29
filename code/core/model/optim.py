# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch
import torch.optim as Optim


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size

    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate 

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * 1):
            r = self.lr_base * 1/4.
        elif step <= int(self.data_size / self.batch_size * 2):
            r = self.lr_base * 2/4.
        elif step <= int(self.data_size / self.batch_size * 3):
            r = self.lr_base * 3/4.
        else:
            r = self.lr_base

        return r


def get_optim(__C, model, data_size, optimizer, lr_base=None):
    if lr_base is None:
        lr_base = __C.LR_BASE

    # modules = model._modules
    # params_list = []
    # for m in modules:
    #     if 'dnc' in m:
    #         params_list.append({
    #             'params': filter(lambda p: p.requires_grad, modules[m].parameters()),
    #             'lr': __C.LR_DNC_BASE,
    #             'flag': True
    #         })
    #     else:
    #         params_list.append({
    #             'params': filter(lambda p: p.requires_grad, modules[m].parameters()),

    #         })
    if optimizer == 'adam':
        optim = Optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0,
                betas=__C.OPT_BETAS,
                eps=__C.OPT_EPS,

            )
    elif optimizer == 'rmsprop':
        optim = Optim.RMSprop(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0,
                eps=__C.OPT_EPS,
                weight_decay=__C.OPT_WEIGHT_DECAY
            )
    else:
        raise ValueError('{} optimizer is not supported'.fromat(optimizer))
    return WarmupOptimizer(
        lr_base,
        optim,
        data_size,
        __C.BATCH_SIZE
    )


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r

def adjust_lr_dnc(optim, decay_r):
    optim.lr_dnc_base *= decay_r
