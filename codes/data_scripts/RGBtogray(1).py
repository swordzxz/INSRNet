import os
import numpy as np
import cv2


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])  # 分别对应通道 R G B
def main():
    # input_path = '/media/omnisky/ubuntu/zxz/model/EDSR/test_image/result/DIV2K100HRx2-Demo'
    # output_path = '/media/omnisky/ubuntu/zxz/model/EDSR/test_image/result/DIV2K100HRx2-Demo'
    input_path = r'/media/omnisky/ubuntu/zxz/obj_data/DIV2K/EC/train/LRRGB'
    output_path = r'/media/omnisky/ubuntu/zxz/obj_data/DIV2K/EC/train/LR'
    dirs = os.listdir(input_path)
    for dir in dirs:
        # image = rgb2gray(cv2.imread(input_path+'/'+dir))
        image = cv2.imread(input_path+'/'+dir)
        image = image[:, :, 0]
        cv2.imwrite((output_path+'/'+dir), image)

if __name__ == '__main__':
    main()