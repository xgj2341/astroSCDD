#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name   : scatter_render_main2.py
# Author :  zengsk in NanJing
# Created: 2019/12/11 12:46

from scipy.stats import *
def plot_scatter_teff(x,y,snrg):
    from matplotlib.colors import LogNorm
    from matplotlib import pyplot as plt
    import numpy as np

    xrange = [4000,8000]
    yrange = [4000,8000]
    xticks = np.linspace(4000,8000, 3)
    yticks = np.linspace(4000,8000, 3)
    xlabel = "Teff_LAMOST"
    ylabel_i = "Teff_FC"
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 15}

    # -----------------  Plot Start  -----------------
    plt.figure(1, facecolor="grey")
    # ---------------  sub plot no.1  ----------------

    colormap = plt.get_cmap("jet")  # 色带
    plt.hist2d(x,y,bins=[400,400],norm=LogNorm(),cmap=colormap,)##norm=LogNorm()
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xticks(xticks, fontproperties='Times New Roman', size=15)
    plt.yticks(yticks, fontproperties='Times New Roman', size=15)
    plt.xlabel(xlabel, fontdict=font, size=15)
    plt.ylabel(ylabel_i, fontdict=font, size=15)
    # plt.grid(linestyle='--', color="grey")
    plt.plot(xrange, yrange, color="k", linewidth=2, linestyle='--')
    plt.rc('font', **font)
    # color bar
    colormap = plt.get_cmap("jet")  # 色带
    cbar = plt.colorbar()  # 显示色带
    # cbar.set_label("Scatter Density", fontdict=font, size=15)
    # cbar.set_ticks([1,10,90])
    # cbar.set_ticklabels(('1','10','90'))
    cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=15)  # 主刻度
    cbar.ax.tick_params(which="minor", direction="in", length=0)  # 副刻度
    ##添加文本，保存图像
    if snrg[1] <= 120:
        plt.text(4100, 7600, f'SNRg=[{snrg[0]},{snrg[1]}]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像_lognorm\\teff_{snrg[0]}_{snrg[1]}.png')
    else:
        plt.text(4100, 7600, f'SNRg=[{snrg[0]},+∞]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像_lognorm\\teff_{snrg[0]}_+∞.png')
    plt.close()

def plot_scatter_logg(x,y,snrg):
    from matplotlib.colors import LogNorm
    from matplotlib import pyplot as plt
    import numpy as np

    xrange = [0.5, 5.5]
    yrange = [0.5, 5.5]
    xticks = np.linspace(0.5, 5.5, 3)
    yticks = np.linspace(0.5, 5.5, 3)
    xlabel = "logg_LAMOST"
    ylabel_i = "logg_FC"
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 15}

    # -----------------  Plot Start  -----------------
    plt.figure(1, facecolor="grey")
    # ---------------  sub plot no.1  ----------------

    colormap = plt.get_cmap("jet")  # 色带
    plt.hist2d(x,y,bins=[400,400],norm=LogNorm(),cmap=colormap)##norm=LogNorm()
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xticks(xticks, fontproperties='Times New Roman', size=15)
    plt.yticks(yticks, fontproperties='Times New Roman', size=15)
    plt.xlabel(xlabel, fontdict=font, size=14)
    plt.ylabel(ylabel_i, fontdict=font, size=15)
    # plt.grid(linestyle='--', color="grey")
    plt.plot(xrange, yrange, color="k", linewidth=2, linestyle='--')
    plt.rc('font', **font)
    # color bar
    colormap = plt.get_cmap("jet")  # 色带
    cbar = plt.colorbar()  # 显示色带
    # cbar.set_label("Scatter Density", fontdict=font, size=15)
    # cbar.set_ticks([1,10,90])
    # cbar.set_ticklabels(('1','10','90'))
    cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=15)  # 主刻度
    cbar.ax.tick_params(which="minor", direction="in", length=0)  # 副刻度
    ##添加文本，保存图像
    if snrg[1] <= 120:
        plt.text(0.55, 5, f'SNRg=[{snrg[0]},{snrg[1]}]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像_lognorm\\logg_{snrg[0]}_{snrg[1]}.png')
    else:
        plt.text(0.55, 5, f'SNRg=[{snrg[0]},+∞]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像_lognorm\\logg_{snrg[0]}_+∞.png')
    plt.close()

def plot_scatter_feh(x,y,snrg):
    from matplotlib.colors import LogNorm
    from matplotlib import pyplot as plt
    import numpy as np

    xrange = [-2.5, 1]
    yrange = [-2.5, 1]
    xticks = np.linspace(-2.5, 1, 3)
    yticks = np.linspace(-2.5, 1, 3)
    xlabel = "Feh_LAMOST"
    ylabel_i = "Feh_FC"
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 15}

    # -----------------  Plot Start  -----------------
    plt.figure(1, facecolor="grey")
    # ---------------  sub plot no.1  ----------------

    colormap = plt.get_cmap("jet")  # 色带
    plt.hist2d(x,y,bins=[400,400],norm=LogNorm(),cmap=colormap,)##norm=LogNorm()
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xticks(xticks, fontproperties='Times New Roman', size=15)
    plt.yticks(yticks, fontproperties='Times New Roman', size=15)
    plt.xlabel(xlabel, fontdict=font, size=14)
    plt.ylabel(ylabel_i, fontdict=font, size=14)
    # plt.grid(linestyle='--', color="grey")
    plt.plot(xrange, yrange, color="k", linewidth=2, linestyle='--')
    plt.rc('font', **font)
    # color bar
    colormap = plt.get_cmap("jet")  # 色带
    cbar = plt.colorbar()  # 显示色带
    # cbar.set_label("Scatter Density", fontdict=font, size=15)
    # cbar.set_ticks([1,10,90])
    # cbar.set_ticklabels(('1','10','90'))
    cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=15)  # 主刻度
    cbar.ax.tick_params(which="minor", direction="in", length=0)  # 副刻度
    ##添加文本，保存图像
    if snrg[1] <= 120:
        plt.text(-2.3, 0.65, f'SNRg=[{snrg[0]},{snrg[1]}]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像_lognorm\\feh_{snrg[0]}_{snrg[1]}.png')
    else:
        plt.text(-2.3, 0.65, f'SNRg=[{snrg[0]},+∞]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像_lognorm\\feh_{snrg[0]}_+∞.png')
    plt.close()

def plot_hist(x):

    import seaborn as sns
    import matplotlib.pyplot as plt

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    sns.set_style('darkgrid')
    sns.set_context('paper')
    f, ax = plt.subplots()
    sns.distplot(x,norm_hist=True, color = 'b')

    xrange = [-2, 2]
    plt.xlim(xrange)
    xticks = np.linspace(-2, 2, 5)
    plt.xticks(  xticks,fontproperties='Times New Roman', size=14)
    plt.yticks( fontproperties='Times New Roman', size=14)

    ax.set_xlabel(r'$\Delta$ logg', fontproperties='Times New Roman',fontsize=13)
    ax.set_ylabel('Density', fontproperties='Times New Roman',fontsize=14)
    plt.savefig(f'E:\项目\参数测量\实验一\同一\hist\\hist_logg.png')

def caculate_num(x,y):
    num_all = len(y)
    x = np.asarray(x)
    y = np.asarray(y)
    diff = y - x
    bins = 8
    step = (max(diff) - min(diff)) / bins
    x_left = np.linspace(-(bins/2+0.5)*step,0.5*step,bins/2+2)
    x_right = np.linspace(0.5*step,(bins/2+0.5)*step,10,bins/2+1)
    x = np.hstack((x_left,x_right))

    x_temp = []
    y_temp = []
    for i in range(len(x)-1):
        x_temp.extend(np.linspace(x[i], x[i+1], 5))
        num = len(np.where((diff >= x[i]) & (diff < x[i+1]))[0])/num_all
        y_temp.extend([num] * 5)
    return x_temp,y_temp

def plot_plot():
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    data = pd.read_csv('E:\项目\参数测量\实验一\同一/dr6_50w_snrg_feh.csv')
    # data = data.iloc[0:20000]
    x = np.asarray(data['feh_dr6_test'])  # 观测数据
    y = np.asarray(data['feh_pre']) # 预测数据
    # plot_hist(y-x)
    x_A = []
    x_F = []
    x_G = []
    x_K = []
    y_A = []
    y_F = []
    y_G = []
    y_K = []

    for i,row in enumerate(data['subclass']):
        row = row[2]
        if row == 'A':
            x_A.append(x[i])
            y_A.append(y[i])
        elif row =='F':
            x_F.append(x[i])
            y_F.append(y[i])
        elif row == 'G':
            x_G.append(x[i])
            y_G.append(y[i])
        elif row == 'K':
            x_K.append(x[i])
            y_K.append(y[i])

    from pandas.core.frame import DataFrame

    x_A,y_A = caculate_num(x_A,y_A)
    x_F,y_F = caculate_num(x_F,y_F)
    x_G,y_G = caculate_num(x_G,y_G)
    x_K,y_K = caculate_num(x_K,y_K)

    plt.plot(x_A,y_A)
    plt.plot(x_F,y_F)
    plt.plot(x_G,y_G)
    plt.plot(x_K,y_K)
    plt.show()

import pandas as pd
import numpy as np
data_teff = pd.read_csv('E:\项目\参数测量\实验一\同一/dr6_50w_snrg_teff.csv')
data_logg = pd.read_csv('E:\项目\参数测量\实验一\同一/dr6_50w_snrg_logg.csv')
data_feh = pd.read_csv('E:\项目\参数测量\实验一\同一/dr6_50w_snrg_feh.csv')
# data = data.iloc[0:20000]
y = data_logg['logg_dr6_test']
y_ = data_logg['logg_pre']
plot_hist(y-y_)

snrg_choice = np.asarray([
                          [10,20],
                          [20,40],
                          [40,60],
                          [60,80],
                          [80,100],
                          [100,120],
                          [10,9999999999999999999999999],
                          [120,999999999999999999999]])
snrg = data_teff['snrg_test']
for i in range(8):
    index_snrg = np.where((snrg>=snrg_choice[i,0]) & (snrg<=snrg_choice[i,1]))
    teff = data_teff.iloc[index_snrg]
    logg = data_logg.iloc[index_snrg]
    feh = data_feh.iloc[index_snrg]

    # ##做Herotus_LAMOST
    # x = teff['teff_dr6_test']  # 观测数据
    # y = logg['logg_dr6_test']  # 预测数据
    # x = np.asarray(x).T
    # y = np.asarray(y).T
    # plot_scatter_Herotus_LAMOST(x, y,snrg_choice[i])
    #
    # ##做Herotus_FC
    # x = teff['teff_pre']  # 观测数据
    # y = logg['logg_pre']  # 预测数据
    # x = np.asarray(x).T
    # y = np.asarray(y).T
    # plot_scatter_Herotus_FC(x, y, snrg_choice[i])

    ##做teff
    x = teff['teff_dr6_test']  # 观测数据
    y = teff['teff_pre']  # 预测数据
    x = np.asarray(x).T
    y = np.asarray(y).T
    plot_scatter_teff(x, y, snrg_choice[i])

    ##做logg
    x = logg['logg_dr6_test']  # 观测数据
    y = logg['logg_pre']  # 预测数据
    x = np.asarray(x).T
    y = np.asarray(y).T
    plot_scatter_logg(x, y, snrg_choice[i])

    ##做feh
    x = feh['feh_dr6_test']  # 观测数据
    y = feh['feh_pre']  # 预测数据
    x = np.asarray(x).T
    y = np.asarray(y).T
    plot_scatter_feh(x, y, snrg_choice[i])

    s = 1

