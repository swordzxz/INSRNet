import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import torch
import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import matplotlib.pyplot as plot
from torchvision import utils as utils
import cv2
import os
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # 分别对应通道 R G B
def displayFeature(feature, savepath, image, image_name):
    # plot.axis('off')
    feat_permute = feature.permute(1, 0, 2, 3).cpu()
    if not os.path.exists(savepath+'/'+str(image_name)):
        os.mkdir(savepath+'/'+str(image_name))
    for i in range(64):
        # grid = utils.make_grid(feat_permute.cpu(), nrow=16, normalize=True, padding=10)
        grid = feat_permute[i, :, :, :]
        grid = grid.numpy().transpose((1, 2, 0))[:, :, 0]
        # display_grid = np.zeros(grid.shape)
        display_grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))*255
        display_grid = display_grid.astype(np.uint8)
        # cv2.normalize(grid, display_grid, 0, 255, cv2.NORM_MINMAX)
        # display_grid = ((grid[:, :, 0] + 1)/2)*255
        heatmap = cv2.applyColorMap(display_grid, cv2.COLORMAP_JET)
        # heatmap = cv2.applyColorMap(display_grid, 2)
        cv2.imwrite(savepath+'/'+str(image_name)+'/feature'+str(i)+'.png', heatmap)
        # util.save_img(heatmap, savepath+'/'+str(image_name)+'/feature'+str(i)+'.png')
        # print(savepath+'/'+str(image_name)+'/feature'+str(i)+'.png')
        # fig = plot.gcf()
        # # 去除图像周围的白边
        # height, width = display_grid.shape
        # # 如果dpi=300，那么图像大小=height*width
        # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
        # plot.gca().xaxis.set_major_locator(plot.NullLocator())
        # plot.gca().yaxis.set_major_locator(plot.NullLocator())
        # plot.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plot.margins(0, 0)
        #
        # plot.imshow(display_grid, cmap='jet')
        # # plot.colorbar()
        #
        # plot.savefig(savepath+'/'+str(image_name)+'/'+'feature'+str(i)+'.png', dpi=300)
        # util.save_img(display_grid, savepath+'feature'+str(i)+'.png')
        # merge_image = heatmap_overlay(image, display_grid)
        # plot.imshow(merge_image, cmap='jet')
        # plot.savefig(savepath+'/'+str(image_name)+'/'+'merger'+str(i)+'.png', dpi=300)
def heatmap_overlay(image,heatmap):
    # 灰度化heatmap
    heatmap = heatmap.astype(np.uint8)
    # 热力图伪彩色
    # heatmap_color = cv2.applyColorMap(heatmap_g, cv2.COLORMAP_JET)
    # overlay热力图
    merge_img = image.copy()
    # heatmap_img = heatmap_color.copy()
    overlay = image.copy()
    alpha = 0.3 # 设置覆盖图片的透明度
    # cv2.rectangle(overlay, (0, 0), (merge_img.shape[1], merge_img.shape[0]), (0, 0, 0), -1) # 设置蓝色为热度图基本色
    # cv2.addWeighted(overlay, alpha, merge_img, 1-alpha, 0) # 将背景热度图覆盖到原图
    cv2.addWeighted(merge_img, alpha, heatmap, 1-alpha, 0) # 将热度图覆盖到原图
    return merge_img
#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    if opt['model'] == 'srgan' or opt['model'] =='sr':
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        for data in test_loader:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
            model.feed_data(data, need_GT=need_GT)
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)

            # sr_img = util.tensor2img(visuals['rlt'])  # uint8
            # LQ_img = util.tensor2img(visuals['LQ'])
            sr_img = util.tensor2img(torch.div(torch.add(visuals['rlt'], 1), 2))
            LQ_img = util.tensor2img(torch.div(torch.add(visuals['LQ'], 1), 2))
            # sr_img = cv2.resize(sr_img, (136, 96), interpolation=cv2.INTER_CUBIC)
            # save images
            suffix = opt['suffix']
            if suffix:
                save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = osp.join(dataset_dir, img_name + '.png')
            # LQ_path = osp.join(dataset_dir, img_name +'_LQ'+ '.png')
            # print(LQ_path)
            # util.save_img(LQ_img, LQ_path)

            # calculate PSNR and SSIM
            if need_GT:
                # gt_img = util.tensor2img(visuals['GT'])
                gt_img = util.tensor2img(torch.div(torch.add(visuals['GT'], 1), 2))  # uint8

                sr_img, gt_img = util.crop_border([sr_img[:, :], gt_img[:, :]], 0)
                #GC
                # sr_img, gt_img = util.crop_border([sr_img[:, :], gt_img[:, 4:132]], 0)
                #EC
                # sr_img, gt_img = util.crop_border([sr_img[:, :], gt_img[:, 10:138]], 0)
                util.save_img(sr_img, save_img_path)

                psnr = util.calculate_psnr(sr_img, gt_img)
                ssim = util.calculate_ssim(sr_img, gt_img)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                # if gt_img.shape[2] == 3:  # RGB image
                if gt_img.ndim == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img / 255., only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img / 255., only_y=True)

                    psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
                    ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    logger.info(
                        '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                        format(img_name, psnr, ssim, psnr_y, ssim_y))
                else:
                    logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
            else:
                logger.info(img_name)

        if need_GT:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            # 计算方差
            std_psnr = np.std(test_results['psnr'])
            std_ssim = np.std(test_results['ssim'])
            logger.info(
                '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
                    test_set_name, ave_psnr, ave_ssim))
            logger.info(
                '----Std PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
                    test_set_name, std_psnr, std_ssim))
            if test_results['psnr_y'] and test_results['ssim_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info(
                    '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
                    format(ave_psnr_y, ave_ssim_y))

    elif opt['model'] == 'srgan_joint':
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        G1_dataset_dir = osp.join(opt['path']['results_root'], test_set_name, 'G1')
        G2_dataset_dir = osp.join(opt['path']['results_root'], test_set_name, 'G2')
        util.mkdir(G1_dataset_dir)
        util.mkdir(G2_dataset_dir)

        g1_test_results = OrderedDict()
        g2_test_results = OrderedDict()

        g1_test_results['psnr'] = []
        g1_test_results['ssim'] = []
        g1_test_results['psnr_y'] = []
        g1_test_results['ssim_y'] = []

        g2_test_results['psnr'] = []
        g2_test_results['ssim'] = []
        g2_test_results['psnr_y'] = []
        g2_test_results['ssim_y'] = []
        for data in test_loader:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
            model.feed_data(data, need_GT=need_GT)
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)

            # sr_img = util.tensor2img(visuals['rlt'])  # uint8
            G1_img = util.tensor2img(torch.div(torch.add(visuals['rlt'], 1), 2))
            G2_img = util.tensor2img(torch.div(torch.add(visuals['rlt_2'], 1), 2))  # uint8
            # G2_img = cv2.imread('/media/omnisky/ubuntu/zxz/model/mmsr/val/20191206/val_set14/G2/' + img_name+'.png')
            # G2_img = G2_img[:, :, 0]

            #特征可视化

            # displayFeature(visuals['GAB_inpaint'].view(1, 64, 48, 69),
            #                savepath='/media/omnisky/7D37935326D33C41/zxz/model/mmsr/val/feature/inpaint',
            #                image_name=img_name,
            #                image=G1_img)
            # displayFeature(visuals['GAB_sr'].view(1, 64, 48, 69),
            #                savepath='/media/omnisky/7D37935326D33C41/zxz/model/mmsr/val/feature/sr',
            #                image_name=img_name,
            #                image=G1_img)
            # displayFeature(visuals['fea3'].view(1, 64, 48, 69),
            #                savepath='/media/omnisky/7D37935326D33C41/zxz/model/mmsr/val/feature/fea3',
            #                image_name=img_name,
            #                image=G1_img)
            # displayFeature(visuals['fea5'].view(1, 64, 48, 69),
            #                savepath='/media/omnisky/7D37935326D33C41/zxz/model/mmsr/val/feature/fea5',
            #                image_name=img_name,
            #                image=G1_img)
            # displayFeature(visuals['fea7'].view(1, 64, 48, 69),
            #                savepath='/media/omnisky/7D37935326D33C41/zxz/model/mmsr/val/feature/fea7',
            #                image_name=img_name,
            #                image=G1_img)

            # save images
            suffix = opt['suffix']
            if suffix:
                G1_save_img_path = osp.join(G1_dataset_dir, img_name + suffix + '.png')
                G2_save_img_path = osp.join(G2_dataset_dir, img_name + suffix + '.png')
            else:
                G1_save_img_path = osp.join(G1_dataset_dir, img_name + '.png')
                G2_save_img_path = osp.join(G2_dataset_dir, img_name + '.png')

            util.save_img(G1_img, G1_save_img_path)
            util.save_img(G2_img, G2_save_img_path)
            # calculate PSNR and SSIM
        #     if need_GT:
        #         # gt_img = util.tensor2img(visuals['GT'])
        #         LRgt_img = util.tensor2img(torch.div(torch.add(visuals['LR'], 1), 2))  # uint8
        #         HRgt_img = util.tensor2img(torch.div(torch.add(visuals['GT'], 1), 2))  # uint8
        #
        #         G1_img, LRgt_img = util.crop_border([G1_img, LRgt_img], 0)
        #         G2_img, HRgt_img = util.crop_border([G2_img[:, :], HRgt_img[:, :]], 0)
        #         util.save_img(G1_img, G1_save_img_path)
        #         util.save_img(G2_img, G2_save_img_path)
        #         g1_psnr = util.calculate_psnr(G1_img, LRgt_img)
        #         g1_ssim = util.calculate_ssim(G1_img, LRgt_img)
        #         g2_psnr = util.calculate_psnr(G2_img, HRgt_img)
        #         g2_ssim = util.calculate_ssim(G2_img, HRgt_img)
        #         g1_test_results['psnr'].append(g1_psnr)
        #         g1_test_results['ssim'].append(g1_ssim)
        #         g2_test_results['psnr'].append(g2_psnr)
        #         g2_test_results['ssim'].append(g2_ssim)
        #         # if gt_img.shape[2] == 3:  # RGB image
        #         if LRgt_img.ndim == 3:  # RGB image
        #             sr_img_y = bgr2ycbcr(sr_img / 255., only_y=True)
        #             gt_img_y = bgr2ycbcr(gt_img / 255., only_y=True)
        #
        #             psnr_y = util.calculate_psnr(sr_img_y * 255, gt_img_y * 255)
        #             ssim_y = util.calculate_ssim(sr_img_y * 255, gt_img_y * 255)
        #             test_results['psnr_y'].append(psnr_y)
        #             test_results['ssim_y'].append(ssim_y)
        #             logger.info(
        #                 'RGB:{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
        #                     format(img_name, psnr, ssim, psnr_y, ssim_y))
        #         else:
        #             logger.info('{:20s} - G1 - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, g1_psnr, g1_ssim))
        #             logger.info('{:20s} - G2 - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, g2_psnr, g2_ssim))
        #     else:
        #         logger.info(img_name)
        #
        # if need_GT:  # metrics
        #     # Average PSNR/SSIM results
        #     g1_ave_psnr = sum(g1_test_results['psnr']) / len(g1_test_results['psnr'])
        #     g1_ave_ssim = sum(g1_test_results['ssim']) / len(g1_test_results['ssim'])
        #     g2_ave_psnr = sum(g2_test_results['psnr']) / len(g2_test_results['psnr'])
        #     g2_ave_ssim = sum(g2_test_results['ssim']) / len(g2_test_results['ssim'])
        #     # 计算方差
        #     g1_std_psnr = np.std(g1_test_results['psnr'])
        #     g1_std_ssim = np.std(g1_test_results['ssim'])
        #     g2_std_psnr = np.std(g2_test_results['psnr'])
        #     g2_std_ssim = np.std(g2_test_results['ssim'])
        #     logger.info('------------------G1-----------------')
        #     logger.info(
        #         '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
        #             test_set_name, g1_ave_psnr, g1_ave_ssim))
        #     logger.info(
        #         '----Std PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
        #             test_set_name, g1_std_psnr, g1_std_ssim))
        #     logger.info('------------------G2-----------------')
        #     logger.info(
        #         '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
        #             test_set_name, g2_ave_psnr, g2_ave_ssim))
        #     logger.info(
        #         '----Std PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
        #             test_set_name, g2_std_psnr, g2_std_ssim))
        #     # if test_results['psnr_y'] and test_results['ssim_y']:
        #     #     ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        #     #     ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        #     #     logger.info(
        #     #         '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
        #     #             format(ave_psnr_y, ave_ssim_y))