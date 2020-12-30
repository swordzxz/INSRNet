from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tensorboard.backend.event_processing import event_accumulator
from collections import OrderedDict
import scipy.signal as signal
from scipy import interpolate

def load_event(event_path):
    #加载日志数据
    ea=event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    # print(ea.scalars.Keys())
    logger = OrderedDict()
    for k in ea.scalars.Keys():
        logger[k] = ea.scalars.Items(key=k)
    return logger

def smooth(data,weight):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    # path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        os.makedirs(path)
        # print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False

if __name__ == '__main__':
    # path1 = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200915_1/logger/events.out.tfevents.1600146229.omnisky.21128.0'
    # path2 = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200915_2/logger/events.out.tfevents.1600146418.omnisky.23640.0'
    # path3 = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200917_1/logger/events.out.tfevents.1600329491.omnisky.23232.0'
    # savepath = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/figure/0915_1-0915_2-0917_1/'

    # path1 = '/media/omnisky/ubuntu/zxz/model/mmsr/20200107_2/logger/events.out.tfevents.1578391742.omnisky.23508.0'
    # path2 = '/media/omnisky/ubuntu/zxz/model/mmsr/20200107_1/logger/events.out.tfevents.1578407016.omnisky.24572.0'
    # path3 = '/media/omnisky/ubuntu/zxz/model/mmsr/20200107/logger/events.out.tfevents.1578406975.omnisky.24395.0'
    # savepath = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/figure/0107_2-0107_1-0107/'

    path1 = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200917_1/logger/events.out.tfevents.1600329491.omnisky.23232.0'
    savepath = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/loss/0917_1/'

    logger_1 = load_event(event_path=path1)
    mkdir(savepath)

    np.savetxt(savepath+'l_g1_fea.csv',np.array(logger_1['l_g1_fea']),delimiter=',')
    np.savetxt(savepath+'l_g1_pix.csv',np.array(logger_1['l_g1_pix']),delimiter=',')
    np.savetxt(savepath + 'lr_psnr.csv', np.array(logger_1['lr_psnr']), delimiter=',')
    np.savetxt(savepath + 'l_g2_fea.csv', np.array(logger_1['l_g2_fea']), delimiter=',')
    np.savetxt(savepath + 'l_g2_pix.csv', np.array(logger_1['l_g2_pix']), delimiter=',')
    np.savetxt(savepath + 'hr_psnr.csv', np.array(logger_1['hr_psnr']), delimiter=',')
    print('完成')