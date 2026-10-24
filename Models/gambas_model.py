from __future__ import annotations
import random
import torch
from .base_model import BaseModel
from . import networks2D


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size <= 0: return images
        result = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                result.append(image)
            elif random.random() > 0.5:
                idx = random.randrange(self.pool_size)
                result.append(self.images[idx].clone())
                self.images[idx] = image
            else:
                result.append(image)
        return torch.cat(result, dim=0)


class GambasModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.loss_D = torch.tensor(0.0)
        self.loss_G_GAN = torch.tensor(0.0)
        self.loss_G_perc = torch.tensor(0.0)

    def name(self): return 'GambasModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.lambda_adv = float(getattr(opt, 'lambda_adv', 0.0))
        self.lambda_perc = float(getattr(opt, 'lambda_perc', 0.0))
        self.use_gan = self.isTrain and self.lambda_adv > 0
        self.loss_names = ['D', 'G_GAN', 'G_L1', 'G_perc']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G'] + (['D'] if self.use_gan else [])

        self.netG = networks2D.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
            img_size=(opt.imgH, opt.imgW)
        ).to(self.device)

        if self.use_gan:
            self.netD = networks2D.define_D(
                opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                opt.n_layers_D, opt.norm, False, opt.init_type,
                opt.init_gain, self.gpu_ids
            ).to(self.device)

        if not self.isTrain or getattr(opt, 'continue_train', False):
            self.load_networks(opt.which_epoch)

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G]
            if self.use_gan:
                self.fake_AB_pool = ImagePool(opt.pool_size)
                self.criterionGAN = networks2D.GANLoss(
                    use_lsgan=not opt.no_lsgan,
                    target_real_label=0.9,
                    target_fake_label=0.0,
                ).to(self.device)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr * 0.25, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input_pair):
        a_to_b = self.opt.which_direction == 'AtoB'
        self.real_A = input_pair[0 if a_to_b else 1].to(self.device)
        self.real_B = input_pair[1 if a_to_b else 0].to(self.device)

    def forward(self): self.fake_B = self.netG(self.real_A)

    def backward_G(self):
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G_GAN = torch.zeros((), device=self.device)
        self.loss_G_perc = torch.zeros((), device=self.device)
        total = self.loss_G_L1
        if self.use_gan:
            pred_fake = self.netD(torch.cat((self.real_A, self.fake_B), dim=1))
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            total = total + self.lambda_adv * self.loss_G_GAN
        self.loss_G = total
        self.loss_G.backward()

    def backward_D(self):
        fake_pair = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), dim=1))
        pred_fake = self.netD(fake_pair.detach())
        pred_real = self.netD(torch.cat((self.real_A, self.real_B), dim=1))
        self.loss_D = 0.5 * (self.criterionGAN(pred_fake, False) + self.criterionGAN(pred_real, True))
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        if self.use_gan: self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad(set_to_none=True)
        self.backward_G()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=5.0)
        self.optimizer_G.step()
        if self.use_gan:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad(set_to_none=True)
            self.backward_D()
            self.optimizer_D.step()
        else:
            self.loss_D = torch.zeros((), device=self.device)
