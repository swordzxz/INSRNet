import models.archs.RRDBNet_arch as RRDBNet_arch
from thop import profile
import options.options as option
import argparse
import torch
if __name__ == '__main__':
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    ##model
    input = torch.randn(1, int(opt['network_G']['in_nc']),
                        int(opt['datasets']['train']['GT_size']/opt['scale']),
                        int(opt['datasets']['train']['GT_size']/opt['scale']))
    flops, params = profile(RRDBNet_arch.RRDBNet(in_nc=opt['network_G']['in_nc'], out_nc=opt['network_G']['out_nc'],
                                    nf=opt['network_G']['nf'], nb=opt['network_G']['nb']), inputs=(input,))
    print('Flops:', '%5e'%flops, 'params:', '%5e'%params)
