import os
import cv2

input_path = '/media/omnisky/ubuntu/zxz/DIV2K/DIV2k800/HR'
# mask = cv2.imread('/media/omnisky/ubuntu/zxz/data/mask/mask.png', cv2.COLOR_BGR2GRAY)
output_path = '/media/omnisky/ubuntu/zxz/DIV2K/DIV2k800/GC'
dirs = os.listdir(input_path)
for dir in dirs:
    img=cv2.imread(input_path+'/'+dir)
    cv2.imwrite(output_path + '/' + dir, img[:, 4:132, 0])
