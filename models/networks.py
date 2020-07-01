import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler

from pdb import set_trace as ST

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, label_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'attention_resnet':
        netG = AttentionResnetGenerator(input_nc, output_nc, label_nc, ngf, n_blocks=6)
    elif which_model_netG == 'no_attention_resnet':
        netG = NOAttentionResnetGenerator(input_nc, output_nc, label_nc, ngf, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


def define_D(input_size, input_nc, label_nc, ndf, which_model_netD,
             n_layers_D=6, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'attention_PatchGAN':
        netD = AttentionPatchGANDiscriminator(input_size, ndf, label_nc, n_layers=n_layers_D)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)

def mask_activation_loss(x):
    return torch.mean(x)

def mask_smooth_loss(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# Defines the generator that contains an attention module
# Several differences from G used in cycleGAN:
#   norm layer: use instance norm by default
#   use bias: do NOT use bias when doing convolution
class AttentionResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, label_nc, ngf, n_blocks=6):
        assert(n_blocks >= 0)
        super(AttentionResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # initial convolutional layer
        base_model = [nn.Conv2d(input_nc + label_nc, ngf, kernel_size=7, stride=1, padding=3, bias=False),
                 nn.InstanceNorm2d(ngf, affine=True),
                 nn.ReLU(inplace=True)]

        # down-sampling layers
        n_downsampling = 2
        curr_ngf = ngf
        for i in range(n_downsampling):
            base_model += [nn.Conv2d(curr_ngf, curr_ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(curr_ngf * 2, affine=True),
                      nn.ReLU(inplace=True)]
            curr_ngf *= 2

        # bottleneck
        for i in range(n_blocks):
            base_model += [ResidualBlock(input_nc=curr_ngf, output_nc=curr_ngf)]

        # up-sampling layers
        n_upsampling = 2
        for i in range(n_upsampling):
            base_model += [nn.ConvTranspose2d(curr_ngf, curr_ngf // 2, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(curr_ngf // 2, affine=True),
                      nn.ReLU(inplace=True)]
            curr_ngf = curr_ngf // 2

        self.base_model = nn.Sequential(*base_model)

        # regression layer for generating the attention mask
        A_reg = [nn.Conv2d(curr_ngf, 1, kernel_size=7, stride=1, padding=3, bias=False),
                 nn.Sigmoid()]
        self.A_reg = nn.Sequential(*A_reg)

        # regression layer for generating the main chunk
        C_reg = [nn.Conv2d(curr_ngf, 3, kernel_size=7, stride=1, padding=3, bias=False),
                 nn.Tanh()]
        self.C_reg = nn.Sequential(*C_reg)

    def forward(self, x, label):
        # x:     Float  Tensor of size (N, C, H, W)
        # label: Double Tensor of size (N, C, H ,W)

        x_size = x.size()[-1]
        label = label.expand(-1, -1, x_size, x_size) # expand label vector only along H and W
        x = torch.cat([x, label], dim=1)
        feat = self.base_model(x)
        return self.C_reg(feat), self.A_reg(feat)

# Defines the generator that DOES NOT contain an attention module
# Several differences from G used in cycleGAN:
#   norm layer: use instance norm by default
#   use bias: do NOT use bias when doing convolution
class NOAttentionResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, label_nc, ngf, n_blocks=6):
        assert(n_blocks >= 0)
        super(NOAttentionResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # initial convolutional layer
        base_model = [nn.Conv2d(input_nc + label_nc, ngf, kernel_size=7, stride=1, padding=3, bias=False),
                 nn.InstanceNorm2d(ngf, affine=True),
                 nn.ReLU(inplace=True)]

        # down-sampling layers
        n_downsampling = 2
        curr_ngf = ngf
        for i in range(n_downsampling):
            base_model += [nn.Conv2d(curr_ngf, curr_ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(curr_ngf * 2, affine=True),
                      nn.ReLU(inplace=True)]
            curr_ngf *= 2

        # bottleneck
        for i in range(n_blocks):
            base_model += [ResidualBlock(input_nc=curr_ngf, output_nc=curr_ngf)]

        # up-sampling layers
        n_upsampling = 2
        for i in range(n_upsampling):
            base_model += [nn.ConvTranspose2d(curr_ngf, curr_ngf // 2, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(curr_ngf // 2, affine=True),
                      nn.ReLU(inplace=True)]
            curr_ngf = curr_ngf // 2

        # final layer
        base_model += [nn.Conv2d(curr_ngf, 3, kernel_size=7, stride=1, padding=3, bias=False),
                            nn.Tanh()]

        self.base_model = nn.Sequential(*base_model)

    def forward(self, x, label):
        # x:     Float  Tensor of size (N, C, H, W)
        # label: Double Tensor of size (N, C, H ,W)
        x_size = x.size()[-1]
        label = label.expand(-1, -1, x_size, x_size) # expand label vector only along H and W
        x = torch.cat([x, label], dim=1)
        output =  self.base_model(x)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(ResidualBlock, self).__init__()

        model = [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=False),
                 nn.InstanceNorm2d(output_nc, affine=True),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=False),
                 nn.InstanceNorm2d(output_nc, affine=True)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)

class AttentionPatchGANDiscriminator(nn.Module):
    def __init__(self, input_size, ndf, label_nc, n_layers=6):
        super(AttentionPatchGANDiscriminator, self).__init__()

        base_model = [nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1),
                      nn.LeakyReLU(0.01, inplace=True)]

        curr_ndf = ndf
        for i in range(1, n_layers):
            base_model += [nn.Conv2d(curr_ndf, curr_ndf*2, kernel_size=4, stride=2, padding=1),
                           nn.LeakyReLU(0.01, inplace=True)]
            curr_ndf *= 2

        output_size = int(input_size / np.power(2, n_layers))
        self.base_model = nn.Sequential(*base_model)
        self.conv_heatmap = nn.Conv2d(curr_ndf, 1, kernel_size=3, stride=1, padding=1, bias=False) # (N, 1, H, W), dense likelihood heatmap of PatchGAN
        self.conv_reglabel = nn.Conv2d(curr_ndf, label_nc, kernel_size=output_size, bias=False)    # (N, C, 1, 1), predicted label vector

    def forward(self, x):
        feat = self.base_model(x)
        out_featmap = self.conv_heatmap(feat)
        out_reglabel = self.conv_reglabel(feat)
        return out_featmap.squeeze(), out_reglabel.squeeze()

#################################################
### BELOW: Legacy Code from Original CycleGAN ###
#################################################

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

##########################################################
# LightCNN-9Layers for identity information preservation
##########################################################
class lightCNN_9layers(nn.Module):
    def __init__(self, input_nc=3, gpu_ids=[]):
        super(lightCNN_9layers, self).__init__()
        self.gpu_ids = gpu_ids

        self.resize_layer = nn.Upsample(size=(128,128), mode='bilinear')
        self.features = nn.Sequential(
            mfm(input_nc, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)

    def forward(self, x):
        x = self.resize_layer(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def load_pretrained_model(self, params_path='../../pretrained_models/LightCNN/LightCNN_9layers/LightCNN_9Layers_checkpoint.pth.tar'):
        ST()
        data = torch.load(params_path)
        pretrained_state_dict = data['state_dict']

        # load parameters for self.features
        features_state_dict_new = {}
        for key in self.features.state_dict().keys():
            features_state_dict_new[key] = pretrained_state_dict['module.features.' + key]
        self.features.load_state_dict(features_state_dict_new)

        for params in self.features.parameters():
            params.requires_grad = False

        # load parameters for self.fc1
        fc1_state_dict_new = {}
        for key in self.fc1.state_dict().keys():
            fc1_state_dict_new[key] = pretrained_state_dict['module.fc1.' + key]
        self.fc1.load_state_dict(fc1_state_dict_new)

        for params in self.fc1.parameters():
            params.requires_grad = False

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=1)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x