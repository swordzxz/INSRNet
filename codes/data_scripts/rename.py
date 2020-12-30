import os
import glob


def main():
    folder = '/media/omnisky/ubuntu/zxz/model/EDSR/test_image/result/BSDS100HRx2-Demo'
    DIV2K(folder)
    print('Finished.')


def DIV2K(path):
    img_path_l = glob.glob(os.path.join(path, '*'))
    for img_path in img_path_l:
        new_path = img_path.replace('_x4_SR.png', '.png').replace('', '').replace('', '').replace('', '')
        os.rename(img_path, new_path)


if __name__ == "__main__":
    main()