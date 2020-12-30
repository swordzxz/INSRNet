import numpy as np

import glob
import os
import threading
from PIL import Image

input_images_path = "/media/omnisky/ubuntu/zxz/newdata/DIV2K/test/HRx2"
output_images_path = "/media/omnisky/ubuntu/zxz/newdata/DIV2K/test/HRRGBx2"


# 将读取到的文件保存到指定文件夹中
# def create_image(infile, index, dir):
#     os.path.splitext(infile)
#     im = Image.open(infile)
#     image = np.expand_dims(im, axis=2)
#     image = np.concatenate((image, image, image), axis=-1)
#     image = Image.fromarray(image)
#     image.save(output_images_path + "/" + str(dir),'PNG')  # 存储路径

#####重命名
def create_image(infile, index, dir):
    os.path.splitext(infile)
    image = Image.open(infile)
    image = np.expand_dims(image, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    dir_cut = dir[:-4]
    # print(dir_cut)
    image.save(output_images_path + "/" + str(dir_cut)+".png",'PNG')  # 存储路径

# 读取文件夹中的全部图片
def start():
    dirs = os.listdir(input_images_path)
    if not os.path.exists(output_images_path):
        os.mkdir(output_images_path)
    for dir in dirs:
        for index in range(1):
            for infile in glob.glob(input_images_path + "/" +dir):  # 数据来源

                print(dir)
                t = threading.Thread(target=create_image, args=(infile, index, dir))
                t.start()
                t.join()
                index += 1



if __name__ == "__main__":
    start()
    print('ok')
