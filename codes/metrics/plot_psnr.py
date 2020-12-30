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

    path1 = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200918_1/logger/events.out.tfevents.1600432552.omnisky.6411.0'
    path2 = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200918_2/logger/events.out.tfevents.1600432900.omnisky.11381.0'
    path3 = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200917_1/logger/events.out.tfevents.1600329491.omnisky.23232.0'
    savepath = '/media/omnisky/7D37935326D33C41/zxz/model/mmsr/figure/0918_1-0918_2-0917_1/'

    logger_1 = load_event(event_path=path1)
    logger_2 = load_event(event_path=path2)
    logger_3 = load_event(event_path=path3)
    mkdir(savepath)

    logger_1['l_g_total'] = []
    logger_2['l_g_total'] = []
    logger_3['l_g_total'] = []
    ymin = 15
    ymax = 30
    filter_size = 0.9
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    x1 = [i.step for i in logger_1['psnr']]
    y1 = [i.value for i in logger_1['psnr']]
    list_1 = list(zip(x1, y1))
    y1 = smooth(y1, weight=filter_size)

    x2 = [i.step for i in logger_2['hr_psnr']]
    y2 = [i.value for i in logger_2['hr_psnr']]
    list_2 = list(zip(x2, y2))
    y2 = smooth(y2, weight=filter_size)

    x3 = [i.step for i in logger_3['hr_psnr']]
    y3 = [i.value for i in logger_3['hr_psnr']]
    list_3 = list(zip(x3, y3))
    y3 = smooth(y3, weight=filter_size)
    np.savetxt(savepath+'SR.csv',np.array(list_1),delimiter=',')
    np.savetxt(savepath+'SR-IN.csv',np.array(list_2),delimiter=',')
    np.savetxt(savepath+'SR-F-IN.csv',np.array(list_3),delimiter=',')

    # ax1.plot(x1, y1, 'b-', label=u'SR')
    # ax1.plot(x2, y2, 'r-', label=u'SR-IN')
    # ax1.plot(x3, y3, 'k-', label=u'SR-F-IN')
    #
    #
    # title = 'compare psnr'
    # ylabel = 'psnr'
    # ax1.legend(loc='upper right', fontsize='x-large')
    # ax1.set_title(str(title), fontsize=16)
    # ax1.set_xlabel(u'Step', fontsize=16)
    # ax1.set_ylabel(str(ylabel), fontsize=16)
    # ax1.tick_params(axis='both', which='major', labelsize=10)
    #
    # plt.ylim(ymin, ymax)
    # # plt.xlim(0, 410000)
    # fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    # plt.savefig(savepath + '/' + str(title) + '.png')
    # plt.show()

    print('绘制完成')