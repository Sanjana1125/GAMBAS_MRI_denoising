from __future__ import annotations
import functools
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from . import mamba_modules2D


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    if norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == 'none':
        return None
    raise NotImplementedError(norm_type)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        fn = lambda epoch: max(0.0, 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1))
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
    if opt.lr_policy == 'identity':
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if opt.lr_policy == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    if opt.lr_policy == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    if opt.lr_policy == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    raise NotImplementedError(opt.lr_policy)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        name = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in name or 'Linear' in name):
            if init_type == 'normal': init.normal_(m.weight, 0.0, gain)
            elif init_type == 'xavier': init.xavier_normal_(m.weight, gain=gain)
            elif init_type == 'kaiming': init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal': init.orthogonal_(m.weight, gain=gain)
            else: raise NotImplementedError(init_type)
            if getattr(m, 'bias', None) is not None: init.constant_(m.bias, 0.0)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    gpu_ids = gpu_ids or []
    if gpu_ids and torch.cuda.is_available():
        net.to(torch.device(f'cuda:{gpu_ids[0]}'))
        if len(gpu_ids) > 1: net = nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False,
             init_type='normal', init_gain=0.02, gpu_ids=None, **kwargs):
    if netG in ('resnet', 'resnet_2d', 'plain_resnet'):
        net = mamba_modules2D.ResNet2DGenerator(input_nc, output_nc, ngf=ngf, sequence_backend=None)
    elif netG in ('gambas', 'gambas_gru', 'hilbert_gru'):
        net = mamba_modules2D.ResNet2DGenerator(input_nc, output_nc, ngf=ngf, sequence_backend='gru')
    elif netG in ('gambas_identity', 'hilbert_identity'):
        net = mamba_modules2D.ResNet2DGenerator(input_nc, output_nc, ngf=ngf, sequence_backend='identity')
    elif netG in ('gambas_pure_mamba', 'pure_mamba'):
        net = mamba_modules2D.ResNet2DGenerator(input_nc, output_nc, ngf=ngf, sequence_backend='pure_mamba')
    else:
        raise NotImplementedError(f'Unknown generator {netG}')
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()

    def forward(self, prediction, target_is_real):
        label = self.real_label if target_is_real else self.fake_label
        return self.loss(prediction, label.expand_as(prediction))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        use_bias = True
        kw, pad = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, 2, pad), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 2, pad, bias=use_bias),
                    norm_layer(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 1, pad, bias=use_bias),
                norm_layer(ndf * mult), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * mult, 1, kw, 1, pad)]
        self.model = nn.Sequential(*seq)

    def forward(self, x): return self.model(x)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='instance', use_sigmoid=False,
             init_type='normal', init_gain=0.02, gpu_ids=None):
    norm_layer = get_norm_layer(norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers_D if netD == 'n_layers' else 3, norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)
