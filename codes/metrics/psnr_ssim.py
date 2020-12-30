'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
# import utils.util as util
from collections import OrderedDict
import logging


def main():
    # Configurations
    logger = logging.getLogger('base')
    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = r'C:\Users\Administrator\Desktop\7x7\test\HR'
    folder_Gen = r'C:\Users\Administrator\Desktop\DIV2K\model\test\20200327\bicubic'

    suffix = ''  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    img_list = sorted(glob.glob(folder_GT + '/*'))

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')
    g2_test_results = OrderedDict()
    g2_test_results['psnr'] = []
    g2_test_results['ssim'] = []
    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        im_GT = cv2.imread(img_path) / 255.
        im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + suffix + '.png')) / 255.

        # if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        #     im_GT_in = bgr2ycbcr(im_GT)
        #     im_Gen_in = bgr2ycbcr(im_Gen)
        # else:

        # calculate PSNR and SSIM
        # PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)
        #
        # SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
        g2_psnr = calculate_psnr(im_GT*255, im_Gen*255)
        g2_ssim = calculate_ssim(im_GT*255, im_Gen*255)
        g2_test_results['psnr'].append(g2_psnr)
        g2_test_results['ssim'].append(g2_ssim)

        print('{:20s} - G2 - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(base_name, g2_psnr, g2_ssim))
    g2_ave_psnr = sum(g2_test_results['psnr']) / len(g2_test_results['psnr'])
    g2_ave_ssim = sum(g2_test_results['ssim']) / len(g2_test_results['ssim'])
    g2_std_psnr = np.std(g2_test_results['psnr'])
    g2_std_ssim = np.std(g2_test_results['ssim'])
    print('------------------G2-----------------')
    test_set_name ='DIV2K'
    print(
        '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
            test_set_name, g2_ave_psnr, g2_ave_ssim))
    print(
        '----Std PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
            test_set_name, g2_std_psnr, g2_std_ssim))
        # print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
        #     i + 1, base_name, PSNR, SSIM))
        # PSNR_all.append(PSNR)
        # SSIM_all.append(SSIM)
    # print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
    #     sum(PSNR_all) / len(PSNR_all),
    #     sum(SSIM_all) / len(SSIM_all)))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
