import skimage
import io,os
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance,ImageChops
import cv2
import numpy as np
import random
#root_path为图像根目录，img_name为图像名字

def move(root_path,img_name,off): #平移，平移尺度为off
    img = Image.open(os.path.join(root_path, img_name))
    offset = ImageChops.offset(img,off,0)
    # offset = img.offset(off,0)
    return offset

def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '.png'))
    return filp_img

def aj_contrast(root_path,img_name): #调整对比度 两种方式 gamma/log
    image = skimage.io.imread(os.path.join(root_path, img_name))
    # gam = skimage.exposure.adjust_gamma(image, 0.5)
    # skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_gam.jpg'),gam)
    log= skimage.exposure.adjust_log(image)
    # skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_log.jpg'),log)
    return log
def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(90*3) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def randomGaussian(root_path, img_name, mean, sigma):  #高斯噪声
    image = Image.open(os.path.join(root_path, img_name))
    im = np.array(image)
    #设定高斯函数的偏移
    means = 0
    #设定高斯函数的标准差
    sigma = 25
    #r通道
    r = im[:,:,0].flatten()

    #g通道
    g = im[:,:,1].flatten()

    #b通道
    b = im[:,:,2].flatten()

    #计算新的像素值
    for i in range(im.shape[0]*im.shape[1]):

        pr = int(r[i]) + random.gauss(0,sigma)

        pg = int(g[i]) + random.gauss(0,sigma)

        pb = int(b[i]) + random.gauss(0,sigma)

        if(pr < 0):
            pr = 0
        if(pr > 255):
            pr = 255
        if(pg < 0):
            pg = 0
        if(pg > 255):
            pg = 255
        if(pb < 0):
            pb = 0
        if(pb > 255):
            pb = 255
        r[i] = pr
        g[i] = pg
        b[i] = pb
    im[:,:,0] = r.reshape([im.shape[0],im.shape[1]])

    im[:,:,1] = g.reshape([im.shape[0],im.shape[1]])

    im[:,:,2] = b.reshape([im.shape[0],im.shape[1]])
    gaussian_image = gaussian_image = Image.fromarray(np.uint8(im))
    return gaussian_image
def randomColor(root_path, img_name): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # 分别对应通道 R G B
def main():
    input_path = '/media/omnisky/ubuntu/zxz/obj_data/DIV2K/obj_train/GT'
    output_path = '/home/zxz/project/HSGAN/codes/data/GT'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dirs = os.listdir(input_path)
    for dir in dirs:
        print(dir)
        # 翻转
        image = flip(input_path, dir)
        image.save(os.path.join(output_path, dir.split('.')[0] + '_flip_left.png'))
        # 对比度
        image = aj_contrast(input_path, dir)
        skimage.io.imsave(os.path.join(output_path, dir.split('.')[0] + '_aj_log.png'), image)

if __name__ == '__main__':
    main()