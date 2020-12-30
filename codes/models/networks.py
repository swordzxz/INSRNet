import torch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.discriminator_vgg_arch_joint as SRGAN_arch_joint
import models.archs.RRDBNet_arch as RRDBNet_arch
import models.archs.RRDBNet_arch_joint as RRDBNet_arch_joint
import models.archs.AttentiveNet as AttNet_arch
# import models.archs.EDVR_arch as EDVR_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])

    elif which_model == 'AttNet':
        netG = AttNet_arch.AttNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    # video restoration
    # elif which_model == 'EDVR':
    #     netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
    #                           groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
    #                           back_RBs=opt_net['back_RBs'], center=opt_net['center'],
    #                           predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
    #                           w_TSA=opt_net['w_TSA'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF


### 联合网络
def define_G1(opt):
    opt_net = opt['network_G1']
    which_model = opt_net['which_model_G1']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch_joint.RRDBNet1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])

    elif which_model == 'AttNet':
        netG = AttNet_arch.AttNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    # video restoration
    # elif which_model == 'EDVR':
    #     netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
    #                           groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
    #                           back_RBs=opt_net['back_RBs'], center=opt_net['center'],
    #                           predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
    #                           w_TSA=opt_net['w_TSA'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

def define_G2(opt):
    opt_net = opt['network_G2']
    which_model = opt_net['which_model_G2']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch_joint.RRDBNet2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])

    elif which_model == 'AttNet':
        netG = AttNet_arch.AttNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    # video restoration
    # elif which_model == 'EDVR':
    #     netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
    #                           groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
    #                           back_RBs=opt_net['back_RBs'], center=opt_net['center'],
    #                           predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
    #                           w_TSA=opt_net['w_TSA'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

def define_D1(opt):
    opt_net = opt['network_D1']
    which_model = opt_net['which_model_D1']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch_joint.Discriminator_VGG_128_1(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

def define_D2(opt):
    opt_net = opt['network_D2']
    which_model = opt_net['which_model_D2']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch_joint.Discriminator_VGG_128_2(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD