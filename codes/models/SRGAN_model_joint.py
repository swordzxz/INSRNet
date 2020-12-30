import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import GANLoss

logger = logging.getLogger('base')


class SRGANModelJoint(BaseModel):
    def __init__(self, opt):
        super(SRGANModelJoint, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG1 = networks.define_G1(opt).to(self.device)
        self.netG2 = networks.define_G2(opt).to(self.device)
        if opt['dist']:
            self.netG1 = DistributedDataParallel(self.netG1, device_ids=[torch.cuda.current_device()])
            self.netG2 = DistributedDataParallel(self.netG2, device_ids=[torch.cuda.current_device()])
        else:
            self.netG1 = DataParallel(self.netG1)
            self.netG2 = DataParallel(self.netG2)
        # if self.is_train:
        #     self.netD1 = networks.define_D1(opt).to(self.device)
        #     self.netD2 = networks.define_D2(opt).to(self.device)
        #     if opt['dist']:
        #         self.netD1 = DistributedDataParallel(self.netD1, device_ids=[torch.cuda.current_device()])
        #         self.netD2 = DistributedDataParallel(self.netD2, device_ids=[torch.cuda.current_device()])
        #     else:
        #         self.netD1 = DataParallel(self.netD1)
        #         self.netD2 = DataParallel(self.netD2)

            self.netG1.train()
            # self.netD1.train()
            self.netG2.train()
            # self.netD2.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight_1'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix1_w = train_opt['pixel_weight_1']
                self.l_pix2_w = train_opt['pixel_weight_2']

            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight_1'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea1_w = train_opt['feature_weight_1']
                self.l_fea2_w = train_opt['feature_weight_2']

            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    pass  # do not need to use DistributedDataParallel for netF
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan1_w = train_opt['gan_weight_1']
            self.l_gan2_w = train_opt['gan_weight_2']

            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params1 = []
            optim_params2 = []
            for k, v in self.netG1.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params1.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            for k, v in self.netG2.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params2.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G1 = torch.optim.Adam(optim_params1, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizer_G2 = torch.optim.Adam(optim_params2, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            # D
            # wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            # self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(), lr=train_opt['lr_D'],
            #                                     weight_decay=wd_D,
            #                                     betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            # self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=train_opt['lr_D'],
            #                                     weight_decay=wd_D,
            #                                     betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            # # self.optimizers.append(self.optimizer_D1)
            # self.optimizers.append(self.optimizer_D2)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        # self.print_network()  # print network
        self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        self.var_LR = data['LR'].to(self.device)  # LR
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G1
        for p in self.netG1.parameters():
            p.requires_grad = True
        for p in self.netG2.parameters():
            p.requires_grad = False
        # for p in self.netD1.parameters():
        #     p.requires_grad = False
        # for p in self.netD2.parameters():
        #     p.requires_grad = False


        self.optimizer_G1.zero_grad()
        #可视化特征
        self.fake1_H, self.fea = self.netG1(self.var_L)
        # self.fake1_H, self.fea, self.fea3, self.fea5, self.fea7 = self.netG1(self.var_L)

        l_g1_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g1_pix = self.l_pix1_w * self.cri_pix(self.fake1_H, self.var_LR)
                l_g1_total += l_g1_pix
            if self.cri_fea:  # feature loss
                real1_fea = self.netF(self.var_LR).detach()
                fake1_fea = self.netF(self.fake1_H)
                l_g1_fea = self.l_fea1_w * self.cri_fea(fake1_fea, real1_fea)
                l_g1_total += l_g1_fea

            # if self.opt['train']['gan_type'] == 'gan':
            #     pred_g1_fake = self.netD1(self.fake1_H)
            #     l_g1_gan = self.l_gan1_w * self.cri_gan(pred_g1_fake, True)
            # elif self.opt['train']['gan_type'] == 'ragan':
            #     pred_d1_real = self.netD1(self.var_LR).detach()
            #     pred_g1_fake = self.netD1(self.fake1_H)
            #     l_g1_gan = self.l_gan1_w * (
            #         self.cri_gan(pred_d1_real - torch.mean(pred_g1_fake), False) +
            #         self.cri_gan(pred_g1_fake - torch.mean(pred_d1_real), True)) / 2
            # l_g1_total += l_g1_gan

            l_g1_total.backward(retain_graph=True)
            self.optimizer_G1.step()

        # G2
        for p in self.netG1.parameters():
            p.requires_grad = False
        for p in self.netG2.parameters():
            p.requires_grad = True
        # for p in self.netD1.parameters():
        #     p.requires_grad = False
        # for p in self.netD2.parameters():
        #     p.requires_grad = False

        self.optimizer_G2.zero_grad()
        self.fake2_H = self.netG2(self.fake1_H, self.fea)
        # self.fake1_H, self.fake2_H = self.netG2(self.var_L)
        # self.fake2_H, self.GAB_inpaint, self.GAB_sr = self.netG2(self.fake1_H, self.fea)


        # l_g1_total = 0
        # if step % self.D_update_ratio == 0 and step > self.D_init_iters:
        #     if self.cri_pix:  # pixel loss
        #         l_g1_pix = self.l_pix1_w * self.cri_pix(self.fake1_H, self.var_LR)
        #         l_g1_total += l_g1_pix
        #     if self.cri_fea:  # feature loss
        #         real1_fea = self.netF(self.var_LR).detach()
        #         fake1_fea = self.netF(self.fake1_H)
        #         l_g1_fea = self.l_fea1_w * self.cri_fea(fake1_fea, real1_fea)
        #         l_g1_total += l_g1_fea
        #
        #     if self.opt['train']['gan_type'] == 'gan':
        #         pred_g1_fake = self.netD1(self.fake1_H)
        #         l_g1_gan = self.l_gan1_w * self.cri_gan(pred_g1_fake, True)
        #     elif self.opt['train']['gan_type'] == 'ragan':
        #         pred_d1_real = self.netD1(self.var_LR).detach()
        #         pred_g1_fake = self.netD1(self.fake1_H)
        #         l_g1_gan = self.l_gan1_w * (
        #             self.cri_gan(pred_d1_real - torch.mean(pred_g1_fake), False) +
        #             self.cri_gan(pred_g1_fake - torch.mean(pred_d1_real), True)) / 2
        #     l_g1_total += l_g1_gan

        l_g2_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g2_pix = self.l_pix2_w * self.cri_pix(self.fake2_H, self.var_H)
                l_g2_total += l_g2_pix
            if self.cri_fea:  # feature loss
                real2_fea = self.netF(self.var_H).detach()
                fake2_fea = self.netF(self.fake2_H)
                l_g2_fea = self.l_fea2_w * self.cri_fea(fake2_fea, real2_fea)
                l_g2_total += l_g2_fea

            # if self.opt['train']['gan_type'] == 'gan':
            #     pred_g2_fake = self.netD2(self.fake2_H)
            #     l_g2_gan = self.l_gan2_w * self.cri_gan(pred_g2_fake, True)
            # elif self.opt['train']['gan_type'] == 'ragan':
            #     pred_d2_real = self.netD2(self.var_H).detach()
            #     pred_g2_fake = self.netD2(self.fake2_H)
            #     l_g2_gan = self.l_gan2_w * (
            #         self.cri_gan(pred_d2_real - torch.mean(pred_g2_fake), False) +
            #         self.cri_gan(pred_g2_fake - torch.mean(pred_d2_real), True)) / 2
            # l_g2_total += l_g2_gan

            # l_g1_total = 0
            # if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            #     if self.cri_pix:  # pixel loss
            #         l_g1_pix = self.l_pix1_w * self.cri_pix(self.fake1_H, self.var_LR)
            #         l_g1_total += l_g1_pix
            #     if self.cri_fea:  # feature loss
            #         real1_fea = self.netF(self.var_LR).detach()
            #         fake1_fea = self.netF(self.fake1_H)
            #         l_g1_fea = self.l_fea1_w * self.cri_fea(fake1_fea, real1_fea)
            #         l_g1_total += l_g1_fea
            #
            #     # if self.opt['train']['gan_type'] == 'gan':
            #     #     pred_g1_fake = self.netD1(self.fake1_H)
            #     #     l_g1_gan = self.l_gan1_w * self.cri_gan(pred_g1_fake, True)
            #     # elif self.opt['train']['gan_type'] == 'ragan':
            #     #     pred_d1_real = self.netD1(self.var_LR).detach()
            #     #     pred_g1_fake = self.netD1(self.fake1_H)
            #     #     l_g1_gan = self.l_gan1_w * (
            #     #         self.cri_gan(pred_d1_real - torch.mean(pred_g1_fake), False) +
            #     #         self.cri_gan(pred_g1_fake - torch.mean(pred_d1_real), True)) / 2
            #     # l_g1_total += l_g1_gan
            #
            l_g_total = 0.5*l_g2_total+0.5*l_g1_total

            # l_g2_total.backward(retain_graph=True)
            # self.optimizer_G2.step()
            l_g_total.backward(retain_graph=True)
            self.optimizer_G2.step()
            # self.optimizer_G1.step()

        # D1
        # for p in self.netG1.parameters():
        #     p.requires_grad = False
        # for p in self.netG2.parameters():
        #     p.requires_grad = False
        # for p in self.netD1.parameters():
        #     p.requires_grad = True
        # for p in self.netD2.parameters():
        #     p.requires_grad = False
        #
        # self.optimizer_D1.zero_grad()
        # if self.opt['train']['gan_type'] == 'gan':
        #     # need to forward and backward separately, since batch norm statistics differ
        #     # real
        #     pred_d1_real = self.netD1(self.var_LR)
        #     l_d1_real = self.cri_gan(pred_d1_real, True)
        #     l_d1_real.backward()
        #     # fake
        #     pred_d1_fake = self.netD1(self.fake1_H.detach())  # detach to avoid BP to G
        #     l_d1_fake = self.cri_gan(pred_d1_fake, False)
        #     l_d1_fake.backward(retain_graph=True)
        # elif self.opt['train']['gan_type'] == 'ragan':
        #     # pred_d_real = self.netD(self.var_ref)
        #     # pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
        #     # l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
        #     # l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
        #     # l_d_total = (l_d_real + l_d_fake) / 2
        #     # l_d_total.backward()
        #     pred_d1_fake = self.netD1(self.fake1_H.detach()).detach()
        #     pred_d1_real = self.netD1(self.var_LR)
        #     l_d1_real = self.cri_gan(pred_d1_real - torch.mean(pred_d1_fake), True) * 0.5
        #     l_d1_real.backward()
        #     pred_d1_fake = self.netD1(self.fake1_H.detach())
        #     l_d1_fake = self.cri_gan(pred_d1_fake - torch.mean(pred_d1_real.detach()), False) * 0.5
        #     l_d1_fake.backward(retain_graph=True)
        # self.optimizer_D1.step()

        # D2
        # for p in self.netG1.parameters():
        #     p.requires_grad = False
        # for p in self.netG2.parameters():
        #     p.requires_grad = False
        # for p in self.netD1.parameters():
        #     p.requires_grad = False
        # for p in self.netD2.parameters():
        #     p.requires_grad = True
        #
        # self.optimizer_D2.zero_grad()
        # if self.opt['train']['gan_type'] == 'gan':
        #     # need to forward and backward separately, since batch norm statistics differ
        #     # real
        #     pred_d2_real = self.netD2(self.var_H)
        #     l_d2_real = self.cri_gan(pred_d2_real, True)
        #     l_d2_real.backward()
        #     # fake
        #     pred_d2_fake = self.netD2(self.fake2_H.detach())  # detach to avoid BP to G
        #     l_d2_fake = self.cri_gan(pred_d2_fake, False)
        #     l_d2_fake.backward(retain_graph=True)
        # elif self.opt['train']['gan_type'] == 'ragan':
        #     # pred_d_real = self.netD(self.var_ref)
        #     # pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
        #     # l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
        #     # l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
        #     # l_d_total = (l_d_real + l_d_fake) / 2
        #     # l_d_total.backward()
        #     pred_d2_fake = self.netD2(self.fake2_H.detach()).detach()
        #     pred_d2_real = self.netD2(self.var_H)
        #     l_d2_real = self.cri_gan(pred_d2_real - torch.mean(pred_d2_fake), True) * 0.5
        #     l_d2_real.backward()
        #     pred_d2_fake = self.netD2(self.fake2_H.detach())
        #     l_d2_fake = self.cri_gan(pred_d2_fake - torch.mean(pred_d2_real.detach()), False) * 0.5
        #     l_d2_fake.backward(retain_graph=True)
        # self.optimizer_D2.step()


        # # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g1_pix'] = l_g1_pix.item()
                self.log_dict['l_g2_pix'] = l_g2_pix.item()
            if self.cri_fea:
                self.log_dict['l_g1_fea'] = l_g1_fea.item()
                self.log_dict['l_g2_fea'] = l_g2_fea.item()
            # self.log_dict['l_g1_gan'] = l_g1_gan.item()
            # self.log_dict['l_g2_gan'] = l_g2_gan.item()
        #
        # self.log_dict['l_d1_real'] = l_d1_real.item()
        # self.log_dict['l_d1_fake'] = l_d1_fake.item()
        # self.log_dict['D1_real'] = torch.mean(pred_d1_real.detach())
        # self.log_dict['D1_fake'] = torch.mean(pred_d1_fake.detach())

        # self.log_dict['l_d2_real'] = l_d2_real.item()
        # self.log_dict['l_d2_fake'] = l_d2_fake.item()
        # self.log_dict['D2_real'] = torch.mean(pred_d2_real.detach())
        # self.log_dict['D2_fake'] = torch.mean(pred_d2_fake.detach())
    def test(self):
        self.netG1.eval()
        self.netG2.eval()

        with torch.no_grad():
            self.fake1_H, self.fea = self.netG1(self.var_L)
            self.fake2_H = self.netG2(self.fake1_H, self.fea)
            # self.fake1_H,self.fake2_H = self.netG2(self.var_L)
        self.netG1.train()
        self.netG2.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake1_H.detach()[0].float().cpu()
        out_dict['rlt_2'] = self.fake2_H.detach()[0].float().cpu()
        # out_dict['rlt_2'] = self.fake2_H[0].detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
            out_dict['LR'] = self.var_LR.detach()[0].float().cpu()
        # out_dict['GAB_inpaint'] = self.fake2_H[1].detach()[0].float().cpu()
        # out_dict['GAB_sr'] = self.fake2_H[2].detach()[0].float().cpu()
        # 特征可视化
        # out_dict['fea3'] = self.fea3.detach()[0].float().cpu()
        # out_dict['fea5'] = self.fea5.detach()[0].float().cpu()
        # out_dict['fea7'] = self.fea7.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G1 = self.opt['path']['pretrain_model_G1']
        if load_path_G1 is not None:
            logger.info('Loading model for G1 [{:s}] ...'.format(load_path_G1))
            self.load_network(load_path_G1, self.netG1, self.opt['path']['strict_load'])
        # load_path_D1 = self.opt['path']['pretrain_model_D1']
        # if self.opt['is_train'] and load_path_D1 is not None:
        #     logger.info('Loading model for D1 [{:s}] ...'.format(load_path_D1))
        #     self.load_network(load_path_D1, self.netD1, self.opt['path']['strict_load'])
        load_path_G2 = self.opt['path']['pretrain_model_G2']
        if load_path_G2 is not None:
            logger.info('Loading model for G2 [{:s}] ...'.format(load_path_G2))
            self.load_network(load_path_G2, self.netG2, self.opt['path']['strict_load'])
        # load_path_D2 = self.opt['path']['pretrain_model_D2']
        # if self.opt['is_train'] and load_path_D2 is not None:
        #     logger.info('Loading model for D2 [{:s}] ...'.format(load_path_D2))
        #     self.load_network(load_path_D2, self.netD2, self.opt['path']['strict_load'])
    def save(self, iter_step):
        self.save_network(self.netG1, 'G1', iter_step)
        # self.save_network(self.netD1, 'D1', iter_step)
        self.save_network(self.netG2, 'G2', iter_step)
        # self.save_network(self.netD2, 'D2', iter_step)

