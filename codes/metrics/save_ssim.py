import os
import math
import numpy as np
import cv2
import glob
from collections import OrderedDict
import logging


def main():
    folder_GT = r'/media/omnisky/ubuntu/zxz/newdata/DIV2K/test/HR/'
    folder_LR = r'/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200917_1/val_images/G2/'
    all_ssim = []
    for x in range(500,500000,500):

        img_list = sorted(glob.glob(folder_GT + '/*'))
        suffix = ''

        g2_test_results = OrderedDict()
        g2_test_results['psnr'] = []
        g2_test_results['ssim'] = []
        for i, img_path in enumerate(img_list):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # print(base_name)
            im_GT = cv2.imread(img_path) / 255.
            path = folder_LR+base_name+ '/'+ base_name+ '_'+ str(x)+ '.png'
            im_Gen = cv2.imread(path) / 255.


            g2_ssim = calculate_ssim(im_Gen*255, im_GT*255)

            g2_test_results['ssim'].append(g2_ssim)

            g2_ave_ssim = sum(g2_test_results['ssim']) / len(g2_test_results['ssim'])
            print(g2_ave_ssim)
        print(x)
        all_ssim.append(g2_ave_ssim)
    # np.savetxt('/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200917_1/val_images/G2/ssim.csv', all_ssim, delimiter = ',')

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


if __name__ == '__main__':
    main()