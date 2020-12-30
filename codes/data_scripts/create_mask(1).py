import os
import cv2

input_path = '/media/omnisky/ubuntu/zxz/obj_data/DIV2K/obj_test/LR/'
mask = cv2.imread('/media/omnisky/ubuntu/zxz/obj_data/DIV2K/mask.bmp', cv2.COLOR_BGR2GRAY)
output_path = '/media/omnisky/ubuntu/zxz/obj_data/DIV2K/obj_test/mask/'
dirs = os.listdir(input_path)
for dir in dirs:
    # img=cv2.imread(input_path+'/'+dir)
    # cv2.imwrite(output_path + '/' + dir, img[:, 4:132, 0])
    cv2.imwrite(output_path + '/' + dir, mask[:, :])
