from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tensorboard.backend.event_processing import event_accumulator
from collections import OrderedDict
import scipy.signal as signal
from scipy import interpolate

WINDOWS = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

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

def figure_plt_2(logger_1, logger_2, filter_size, data, title, ylabel, ymin, ymax, savepath):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    filter_size = filter_size

    x = [i.step for i in logger_1[str(data)]]
    y = [i.value for i in logger_1[str(data)]]

    y = smooth(y, window_len=10)
    ##learning_rate
    ax1.plot(x, y, 'b-',label=u'Net-1')
    ax1.plot([i.step for i in logger_2[str(data)]][::filter_size], [i.value for i in logger_2[str(data)]][::filter_size], 'r-',label=u'Net-2')

    ax1.legend(loc='upper right')
    ax1.set_title(str(title), fontsize=16)
    ax1.set_xlabel(u'Step', fontsize=16)
    ax1.set_ylabel(str(ylabel), fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    plt.ylim(ymin, ymax)
    plt.xlim(0, 410000)
    print('ymin:', ymin, 'ymax:', ymax)
    fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.savefig(savepath + '/' + str(data) + '.png')
    plt.show()

def figure_plt_3(logger_1, logger_2, logger_3, filter_size, data, title, ylabel, ymin, ymax, savepath):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    if data == 'l_g_total':
        # plot l_g_total
        x1 = [i.step for i in logger_1['l_g_pix']]
        y1 = np.sum([[i.value for i in logger_1['l_g_pix']], [i.value for i in logger_1['l_g_fea']], [i.value for i in logger_1['l_g_gan']]], axis=0)
        y1 = smooth(y1, weight=filter_size)

        x2 = [i.step for i in logger_2['l_g_pix']]
        y2 = np.sum([[i.value for i in logger_2['l_g_pix']], [i.value for i in logger_2['l_g_fea']], [i.value for i in logger_2['l_g_gan']]], axis=0)
        y2 = smooth(y2, weight=filter_size)

        x3 = [i.step for i in logger_3['l_g_pix']]
        y3 = np.sum([[i.value for i in logger_3['l_g_pix']], [i.value for i in logger_3['l_g_fea']], [i.value for i in logger_3['l_g_gan']]], axis=0)
        y3 = smooth(y3, weight=filter_size)

        ax1.plot(x1, y1, 'b-', label=u'Net-1')
        ax1.plot(x2, y2, 'r-', label=u'Net-2')
        ax1.plot(x3, y3, 'k-', label=u'Net-3')

    else:
        x1 = [i.step for i in logger_1[str(data)]]
        y1 = [i.value for i in logger_1[str(data)]]
        y1 = smooth(y1, weight=filter_size)

        x2 = [i.step for i in logger_2[str(data)]]
        y2 = [i.value for i in logger_2[str(data)]]
        y2 = smooth(y2, weight=filter_size)

        x3 = [i.step for i in logger_3[str(data)]]
        y3 = [i.value for i in logger_3[str(data)]]
        y3 = smooth(y3, weight=filter_size)

        ax1.plot(x1, y1, 'b-', label=u'Net-1')
        ax1.plot(x2, y2, 'r-', label=u'Net-2')
        ax1.plot(x3, y3, 'k-', label=u'Net-3')

    ax1.legend(loc='upper right', fontsize='x-large')
    ax1.set_title(str(title), fontsize=16)
    ax1.set_xlabel(u'Step', fontsize=16)
    ax1.set_ylabel(str(ylabel), fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    plt.ylim(ymin, ymax)
    # plt.xlim(0, 410000)
    fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.savefig(savepath + '/' + str(data) + '.png')
    plt.show()

def figure_plt_4(logger_1, logger_2, logger_3, logger_4, filter_size, data, title, ylabel, ymin, ymax, savepath):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    if data == 'l_g_total':
        # plot l_g_total
        x1 = [i.step for i in logger_1['l_g_pix']]
        y1 = np.sum([[i.value for i in logger_1['l_g_pix']], [i.value for i in logger_1['l_g_fea']], [i.value for i in logger_1['l_g_gan']]], axis=0)
        y1 = smooth(y1, weight=filter_size)

        x2 = [i.step for i in logger_2['l_g_pix']]
        y2 = np.sum([[i.value for i in logger_2['l_g_pix']], [i.value for i in logger_2['l_g_fea']], [i.value for i in logger_2['l_g_gan']]], axis=0)
        y2 = smooth(y2, weight=filter_size)

        x3 = [i.step for i in logger_3['l_g_pix']]
        y3 = np.sum([[i.value for i in logger_3['l_g_pix']], [i.value for i in logger_3['l_g_fea']], [i.value for i in logger_3['l_g_gan']]], axis=0)
        y3 = smooth(y3, weight=filter_size)

        x4 = [i.step for i in logger_4['l_g_pix']]
        y4 = np.sum([[i.value for i in logger_4['l_g_pix']], [i.value for i in logger_4['l_g_fea']], [i.value for i in logger_4['l_g_gan']]], axis=0)
        y4 = smooth(y4, weight=filter_size)


        ax1.plot(x1, y1, 'b-', label=u'Net-1')
        ax1.plot(x2, y2, 'r-', label=u'Net-2')
        ax1.plot(x3, y3, 'k-', label=u'Net-3')
        ax1.plot(x4, y4, 'g-', label=u'Net-4')
    else:
        x1 = [i.step for i in logger_1[str(data)]]
        y1 = [i.value for i in logger_1[str(data)]]
        y1 = smooth(y1, weight=filter_size)

        x2 = [i.step for i in logger_2[str(data)]]
        y2 = [i.value for i in logger_2[str(data)]]
        y2 = smooth(y2, weight=filter_size)

        x3 = [i.step for i in logger_3[str(data)]]
        y3 = [i.value for i in logger_3[str(data)]]
        y3 = smooth(y3, weight=filter_size)

        x4 = [i.step for i in logger_4[str(data)]]
        y4 = [i.value for i in logger_4[str(data)]]
        y4 = smooth(y4, weight=filter_size)

        ##learning_rate
        # ax1.plot([i.step for i in logger_1[str(data)]][::filter_size], [i.value for i in logger_1[str(data)][::filter_size]] , 'b-',label=u'Net-1')
        # ax1.plot([i.step for i in logger_2[str(data)]][::filter_size], [i.value for i in logger_2[str(data)]][::filter_size], 'r-',label=u'Net-2')
        # ax1.plot([i.step for i in logger_3[str(data)]][::filter_size], [i.value for i in logger_3[str(data)]][::filter_size], 'k-',label=u'Net-3')
        ax1.plot(x1, y1, 'b-', label=u'Net-1')
        ax1.plot(x2, y2, 'r-', label=u'Net-2')
        ax1.plot(x3, y3, 'k-', label=u'Net-3')
        ax1.plot(x4, y4, 'g-', label=u'Net-4')



    ax1.legend(loc='upper right', fontsize='x-large')
    ax1.set_title(str(title), fontsize=16)
    ax1.set_xlabel(u'Step', fontsize=16)
    ax1.set_ylabel(str(ylabel), fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    plt.ylim(ymin, ymax)
    plt.xlim(0, 410000)
    fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.savefig(savepath + '/' + str(data) + '.png')
    plt.show()

def compare_two(path1, path2, savepath):
    logger_1 = load_event(event_path=path1)
    logger_2 = load_event(event_path=path2)
    y = [i.value for i in logger_1['l_g_pix']][:] + [i.value for i in logger_1['l_g_fea']] + [i.value for i in logger_1['l_g_gan']]
    y1 = np.sum([[i.value for i in logger_1['l_g_pix']], [i.value for i in logger_1['l_g_fea']], [i.value for i in logger_1['l_g_gan']]], axis=0)
    logger_1['l_g_total'] = logger_1['l_g_pix'] + logger_1['l_g_fea'] + logger_1['l_g_gan']
    logger_2['l_g_total'] = logger_2['l_g_pix'] + logger_2['l_g_fea'] + logger_2['l_g_gan']
    mkdir(savepath)
    for k in logger_1.keys():
        print('K:',str(k))
        ymin, ymax, filter_size = label_filter(k)
        print('ymin:', ymin, 'ymax:', ymax)

        figure_plt_2(logger_1, logger_2, filter_size=filter_size, data=k, title=k, ylabel=k, ymin=ymin, ymax=ymax, savepath=savepath)
        print(k+'绘制完成')

def compare_three(path1, path2, path3, savepath):
    logger_1 = load_event(event_path=path1)
    logger_2 = load_event(event_path=path2)
    logger_3 = load_event(event_path=path3)
    # logger_2 = logger_1
    # logger_3 = logger_1
    # logger_4 = logger_1
    logger_1['l_g_total'] = []
    logger_2['l_g_total'] = []
    logger_3['l_g_total'] = []
    k = 'hr_psnr'
    ymin, ymax, filter_size = label_filter(k)
    figure_plt_3(logger_1, logger_2, logger_3, filter_size=filter_size, data=k, title=k, ylabel=k, ymin=ymin, ymax=ymax, savepath=savepath)
    print('绘制完成')
    mkdir(savepath)
    # for k in logger_1.keys():
    #     ymin, ymax, filter_size = label_filter(k)
    #     figure_plt_3(logger_1, logger_2, logger_3, filter_size=filter_size, data=k, title=k, ylabel=k, ymin=ymin, ymax=ymax, savepath=savepath)
    #     print(k+'绘制完成')

def compare_four(path1, path2, path3, path4, savepath):
    logger_1 = load_event(event_path=path1)
    logger_2 = load_event(event_path=path2)
    logger_3 = load_event(event_path=path3)
    logger_4 = load_event(event_path=path4)
    logger_1['l_g_total'] = []
    logger_2['l_g_total'] = []
    logger_3['l_g_total'] = []
    logger_4['l_g_total'] = []
    # logger_2 = logger_1
    # logger_3 = logger_1
    # logger_4 = logger_1
    mkdir(savepath)
    for k in logger_1.keys():
        ymin, ymax, filter_size = label_filter(k)
        figure_plt_4(logger_1, logger_2, logger_3, logger_4, filter_size=filter_size, data=k, title=k, ylabel=k, ymin=ymin, ymax=ymax, savepath=savepath)
        print(k+'绘制完成')

def compare_five(path1, path2, path3, savepath):
    logger_1 = load_event(event_path=path1)
    logger_2 = load_event(event_path=path2)
    logger_3 = load_event(event_path=path3)
    # logger_4 = load_event(event_path=path4)
    # logger_5 = load_event(event_path=path4)

    mkdir(savepath)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    filter_size = 0.5
    # plot l_g_total
    x1 = [i.step for i in logger_1['psnr']]
    y1 = [i.value for i in logger_1['psnr']]
    y1 = smooth(y1, weight=filter_size)

    x2 = [i.step for i in logger_2['hr_psnr']]
    y2 = [i.value for i in logger_2['hr_psnr']]
    y2 = smooth(y2, weight=filter_size)

    x3 = [i.step for i in logger_3['hr_psnr']]
    y3 = [i.value for i in logger_3['hr_psnr']]
    y3 = smooth(y3, weight=filter_size)

    # x4 = [i.step for i in logger_4['psnr']]
    # y4 = [i.value for i in logger_4['psnr']]
    # y4 = smooth(y4, weight=filter_size)
    #
    # x5 = [i.step for i in logger_5['psnr']]
    # y5 = [i.value for i in logger_5['psnr']]
    # y5 = smooth(y5, weight=filter_size)


    ax1.plot(x1, y1, 'b-', label=u'Model_1')
    ax1.plot(x2, y2, 'r-', label=u'Model_2')
    ax1.plot(x3, y3, 'k-', label=u'Model_3')
    # ax1.plot(x4[0:20000], y4[0:20000], 'g-', label=u'8 RRDB 32 filters')
    # ax1.plot(x5[0:20000], y5[0:20000], 'g-', label=u'8 RRDB 128 filters')

    ax1.legend(loc='lower right', fontsize='x-large')
    # ax1.set_title(str(title), fontsize=16)
    ax1.set_xlabel(u'Iters', fontsize=16)
    ax1.set_ylabel(u'PSNR', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    plt.ylim(22, 25)
    plt.xlim(0, 200000)
    # fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
    plt.savefig(savepath + '/' + 'inpainting' + '.png')
    plt.show()

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

def label_filter(k):
    ## 修复
    # if k == 'D_fake' or k == 'D_real':
    #     ymin, ymax = 0, 25000
    #     filter_size = 0.1
    # elif k == 'l_d_fake' or k =='l_d_real':
    #     ymin, ymax = 0, 1.5
    #     filter_size = 0.999
    # elif k == 'l_g_fea':
    #     ymin, ymax = 0, 2
    #     filter_size = 0.99
    # elif k == 'l_g_gan':
    #     ymin, ymax = 0, 0.03
    #     filter_size = 0.999
    # elif k == 'l_g_pix':
    #     ymin, ymax = 0.00125, 0.00075
    #     filter_size = 0.999
    # elif k == 'learning_rate':
    #     ymin, ymax = 0, 0.0001
    #     filter_size = 0
    # elif k == 'psnr':
    #     ymin, ymax = 24, 29
    #     filter_size = 0.5
    # elif k == 'l_g_total':
    #     ymin, ymax =0, 3
    #     filter_size = 0.999
    # 超分辨
    if k == 'D_fake' or k == 'D_real':
        ymin, ymax = 0, 25000
        filter_size = 0.1
    elif k == 'l_d_fake' or k =='l_d_real':
        ymin, ymax = 0, 0.15
        filter_size = 0.999
    elif k == 'l_g_fea':
        ymin, ymax = 0, 2
        filter_size = 0.99
    elif k == 'l_g_gan':
        ymin, ymax = 0, 0.05
        filter_size = 0.999
    elif k == 'l_g_pix':
        ymin, ymax = 0.0008, 0.0005
        filter_size = 0.999
    elif k == 'learning_rate':
        ymin, ymax = 0, 0.0001
        filter_size = 0
    elif k == 'psnr' or k=='hr_psnr':
        ymin, ymax = 15, 30
        filter_size = 0.5
    elif k == 'l_g_total':
        ymin, ymax =0, 3
        filter_size = 0.999
    return ymin, ymax, filter_size
# def box(xmin, xmax, ymin, ymax,)
if __name__ == '__main__':
    # load_event(event_path='/media/omnisky/ubuntu/zxz/model/mmsr/20191118/logger/events.out.tfevents.1574046039.omnisky.29947.0')
    # compare_two(path1='/media/omnisky/ubuntu/zxz/model/mmsr/test/logger/events.out.tfevents.1574652714.omnisky.19839.0',
    #             path2='/media/omnisky/ubuntu/zxz/model/mmsr/test/logger/events.out.tfevents.1574652714.omnisky.19839.0',
    #             savepath='/media/omnisky/ubuntu/zxz/model/mmsr/figure/test/'
    #             )
    compare_three(path1='/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200915_1/logger/events.out.tfevents.1600146229.omnisky.21128.0',
                path2='/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200915_2/logger/events.out.tfevents.1600146418.omnisky.23640.0',
                path3='/media/omnisky/7D37935326D33C41/zxz/model/mmsr/20200917_1/logger/events.out.tfevents.1600329491.omnisky.23232.0',
                savepath='/media/omnisky/7D37935326D33C41/zxz/model/mmsr/figure/0915_1-0915_2-0917_1/'
                )

    # compare_three(path1='/media/omnisky/ubuntu/zxz/model/mmsr/20191114/logger/events.out.tfevents.1573784868.omnisky.8505.0',
    #             path2='/media/omnisky/ubuntu/zxz/model/mmsr/20191107/logger/events.out.tfevents.1573120974.omnisky.23756.0',
    #             path3='/media/omnisky/ubuntu/zxz/model/mmsr/20191120_1/logger/events.out.tfevents.1574387697.omnisky.15752.0',
    #             savepath='/media/omnisky/ubuntu/zxz/model/mmsr/figure/1114-1107-1120_1/'
    #             )
    # compare_three(path1='/media/omnisky/ubuntu/zxz/model/mmsr/20191120_1/logger/events.out.tfevents.1574387697.omnisky.15752.0',
    #             path2='/media/omnisky/ubuntu/zxz/model/mmsr/20191122_2/logger/events.out.tfevents.1574475471.omnisky.10645.0',
    #             path3='/media/omnisky/ubuntu/zxz/model/mmsr/20191120_3/logger/events.out.tfevents.1574388989.omnisky.16406.0',
    #             savepath='/media/omnisky/ubuntu/zxz/model/mmsr/figure/1120_1-1122_2-1120_3/'
    #             )
    # compare_three(path1='/media/omnisky/ubuntu/zxz/model/mmsr/20191120_1/logger/events.out.tfevents.1574387697.omnisky.15752.0',
    #             path2='/media/omnisky/ubuntu/zxz/model/mmsr/20191122_5/logger/events.out.tfevents.1574473784.omnisky.698.0',
    #             path3='/media/omnisky/ubuntu/zxz/model/mmsr/20191122_4/logger/events.out.tfevents.1574474720.omnisky.6521.0',
    #             savepath='/media/omnisky/ubuntu/zxz/model/mmsr/figure/1120_1-1122_5-1122_4/'
    #             )
    # compare_four(path1='/media/omnisky/ubuntu/zxz/model/mmsr/20191120_1_origin/logger/events.out.tfevents.1574233434.omnisky.17465.0',
    #             path2='/media/omnisky/ubuntu/zxz/model/mmsr/20191119_4/logger/events.out.tfevents.1574155505.omnisky.7448.0',
    #             path3='/media/omnisky/ubuntu/zxz/model/mmsr/20191119_1/logger/events.out.tfevents.1574153204.omnisky.18413.0',
    #             path4='/media/omnisky/ubuntu/zxz/model/mmsr/20191120/logger/events.out.tfevents.1574232362.omnisky.19678.0',
    #             savepath='/media/omnisky/ubuntu/zxz/model/mmsr/figure/1120_1-1119_4-1119_1-1120-test/'
    #             )
    # compare_four(path1='/media/omnisky/ubuntu/zxz/model/mmsr/20191122_8/logger/events.out.tfevents.1574599610.omnisky.13695.0',
    #             path2='/media/omnisky/ubuntu/zxz/model/mmsr/20191122_6/logger/events.out.tfevents.1574476081.omnisky.13803.0',
    #             path3='/media/omnisky/ubuntu/zxz/model/mmsr/20191120_1/logger/events.out.tfevents.1574387697.omnisky.15752.0',
    #             path4='/media/omnisky/ubuntu/zxz/model/mmsr/20191122_7/logger/events.out.tfevents.1574578945.omnisky.23699.0',
    #             savepath='/media/omnisky/ubuntu/zxz/model/mmsr/figure/1122_8-1122_6-1120_1-1122_7/'
    #             )

    # #plot the box
    # tx0 = 0
    # tx1 = 10000
    # #设置想放大区域的横坐标范围
    # ty0 = 0.000
    # ty1 = 0.12
    # #设置想放大区域的纵坐标范围
    # sx = [tx0,tx1,tx1,tx0,tx0]
    # sy = [ty0,ty0,ty1,ty1,ty0]
    # plt.plot(sx,sy,"purple")
    # axins = inset_axes(ax1, width=1.5, height=1.5, loc='right')
    # #loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"
    # axins.plot(x1,y1 , color='red', ls='-')
    # axins.plot(x2,y2 , color='blue', ls='-')
    # axins.axis([0,20000,0.000,0.12])
    # plt.savefig("train_results_loss.png")
    #
    # compare_five(path1='/media/omnisky/ubuntu/zxz/model/mmsr/20200107_2/logger/events.out.tfevents.1578391742.omnisky.23508.0',
    #             path2='/media/omnisky/ubuntu/zxz/model/mmsr/20200107_1/logger/events.out.tfevents.1578407016.omnisky.24572.0',
    #             path3='/media/omnisky/ubuntu/zxz/model/mmsr/20200107/logger/events.out.tfevents.1578406975.omnisky.24395.0',
    #             savepath='/media/omnisky/ubuntu/zxz/model/mmsr/figure/ablation/'
    #             )
