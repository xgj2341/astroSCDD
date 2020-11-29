import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import gaussian_kde
import matplotlib.colors as colors
import pandas as pd

def plot_scatter_Herotus_LAMOST(x,y,snrg):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = (z-min(z))/(max(z)-min(z))
    sevp_i = x   # 观测数据
    estimate_i = y # 预测数据


    # ----------- Define Parameters ------------
    # radius = 3  # 半径
    plt.figure()
    colormap = plt.get_cmap("jet")  # 色带
    marker_size = 3  # 散点大小
    xrange = [8000, 4000]
    yrange = [5.5, 0]
    xticks = np.linspace(8000, 4000, 3)
    yticks = np.linspace(5, 0, 3)
    xlabel = "Teff_LAMOST / K"
    ylabel_i = "logg_LAMOST / dex"

    font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 15}
    ###################################plot
    # plt.subplot(1, 1, 1)#aspect="equal"
    plt.scatter(sevp_i, estimate_i, c=z, cmap=colormap, marker=".", s=marker_size)#norm=colors.LogNorm(vmin=z.min(), vmax=0.5 * z.max())
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xticks(xticks, fontproperties='Times New Roman', size=15)
    plt.yticks(yticks, fontproperties='Times New Roman', size=15)

    plt.xlabel(xlabel, fontdict=font, size=15)
    plt.ylabel(ylabel_i, fontdict=font, size=15)
    # plt.grid(linestyle='--', color="grey")
    # plt.plot(xrange, yrange, color="k", linewidth=0.8, linestyle='--')
    plt.rc('font', **font)

    # color bar
    cbar = plt.colorbar()  # 显示色带orientation='horizontal',extend="both", pad=0.1
    # cbar.set_label("Scatter Density", fontdict=font)
    # cbar.set_ticks(np.linspace(0,2*10**(-6),5))
    # cbar.set_ticklabels(('0', '0.5', '1','1.5','2'))
    # cbar.set_ticks()#cbar_ticks
    cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=16)  # 主刻度
    cbar.ax.tick_params(which="minor", direction="in", length=16)  # 副刻度

    ##添加文本
    if snrg[1]<=120:
        plt.text(7900, 0.5, f'SNRg=[{snrg[0]},{snrg[1]}]',fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\Herotus_LAMOST_{snrg[0]}_{snrg[1]}.png')
    else:
        plt.text(7900, 0.5, f'SNRg=[{snrg[0]},+∞]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\Herotus_LAMOST_{snrg[0]}_+∞.png')
    plt.close()

def plot_scatter_Herotus_FC(x,y,snrg):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = (z-min(z))/(max(z)-min(z))
    sevp_i = x   # 观测数据
    estimate_i = y # 预测数据


    # ----------- Define Parameters ------------
    # radius = 3  # 半径
    plt.figure()
    colormap = plt.get_cmap("jet")  # 色带
    marker_size = 3  # 散点大小
    xrange = [8000, 4000]
    yrange = [5.5, 0]
    xticks = np.linspace(8000, 4000, 3)
    yticks = np.linspace(5, 0, 3)
    xlabel = "Teff_FC / K"
    ylabel_i = "logg_FC / dex"

    font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 15}
    ###################################plot
    plt.subplot(1, 1, 1)#aspect="equal"
    plt.scatter(sevp_i, estimate_i, c=z, cmap=colormap, marker=".", s=marker_size)#norm=colors.LogNorm(vmin=z.min(), vmax=0.5 * z.max())
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.xticks(xticks, fontproperties='Times New Roman', size=15)
    plt.yticks(yticks, fontproperties='Times New Roman', size=15)

    plt.xlabel(xlabel, fontdict=font, size=15)
    plt.ylabel(ylabel_i, fontdict=font, size=15)
    # plt.grid(linestyle='--', color="grey")
    # plt.plot(xrange, yrange, color="k", linewidth=0.8, linestyle='--')
    plt.rc('font', **font)

    # color bar
    cbar = plt.colorbar()  # 显示色带orientation='horizontal',extend="both", pad=0.1
    # cbar.set_label("Scatter Density", fontdict=font)
    # cbar.set_ticks(np.linspace(0,2*10**(-6),5))
    # cbar.set_ticklabels(('0', '0.5', '1','1.5','2'))
    # cbar.set_ticks()#cbar_ticks
    cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=15)  # 主刻度
    cbar.ax.tick_params(which="minor", direction="in", length=0)  # 副刻度

    ##添加文本，保存图像
    if snrg[1]<=120:
        plt.text(7900, 0.5, f'SNRg=[{snrg[0]},{snrg[1]}]',fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\Herotus_FC_{snrg[0]}_{snrg[1]}.png')
    else:
        plt.text(7900, 0.5, f'SNRg=[{snrg[0]},+∞]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\Herotus_FC_{snrg[0]}_+∞.png')
    plt.close()

def plot_scatter_teff(x,y,snrg):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = (z - min(z)) / (max(z) - min(z))
    sevp_i = x   # 观测数据
    estimate_i = y # 预测数据


    # ----------- Define Parameters ------------
    # radius = 3  # 半径
    plt.figure()
    colormap = plt.get_cmap("jet")  # 色带
    marker_size = 3  # 散点大小
    xrange = [4000, 8000]
    yrange = [4000, 8000]
    xticks = np.linspace(4000, 8000, 3)
    yticks = np.linspace(4000, 8000, 3)
    xlabel = "Teff_LAMOST / K"
    ylabel_i = "Teff_FC / K"
    ylabel_ii = "Estimate-2"
    ylabel_iii = "Estimate-3"
    # cbar_ticks = [10**0, 10**1, 10**2, 10**3, 10**4, 10**5]
    font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 16}
    ###################################plot
    plt.subplot(1, 1, 1, aspect="equal")
    plt.scatter(sevp_i, estimate_i, c=z, cmap=colormap, marker=".", s=marker_size)#norm=colors.LogNorm(vmin=z.min(), vmax=0.5 * z.max())
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
    cbar = plt.colorbar()  # 显示色带orientation='horizontal',extend="both", pad=0.1
    # cbar.set_label("Scatter Density", fontdict=font)
    # cbar.set_ticks(np.linspace(0,2*10**(-6),5))
    # cbar.set_ticklabels(('0', '0.5', '1','1.5','2'))
    # cbar.set_ticks()#cbar_ticks
    cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=15)  # 主刻度
    cbar.ax.tick_params(which="minor", direction="in", length=0)  # 副刻度

    ##添加文本
    ##添加文本，保存图像
    if snrg[1] <= 120:
        plt.text(4100, 7600, f'SNRg=[{snrg[0]},{snrg[1]}]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\teff_{snrg[0]}_{snrg[1]}.png')
    else:
        plt.text(4100, 7600, f'SNRg=[{snrg[0]},+∞]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\teff_{snrg[0]}_+∞.png')
    plt.close()

def plot_scatter_logg(x,y,snrg):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = (z - min(z)) / (max(z) - min(z))
    sevp_i = x   # 观测数据
    estimate_i = y # 预测数据


    # ----------- Define Parameters ------------
    # radius = 3  # 半径
    colormap = plt.get_cmap("jet")  # 色带
    marker_size = 3  # 散点大小
    xrange = [0.5, 5]
    yrange = [0.5, 5]
    xticks = np.linspace(0.5, 5.5, 3)
    yticks = np.linspace(0.5, 5.5, 3)
    xlabel = "logg_LAMOST / dex"
    ylabel_i = "logg_FC / dex"

    font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 15}
    ###################################plot
    plt.figure()
    plt.subplot(1, 1, 1, aspect="equal")
    plt.scatter(sevp_i, estimate_i, c=z, cmap=colormap, marker=".", s=marker_size)#norm=colors.LogNorm(vmin=z.min(), vmax=0.5 * z.max())
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
    cbar = plt.colorbar()  # 显示色带orientation='horizontal',extend="both", pad=0.1
    # cbar.set_label("Scatter Density", fontdict=font)
    # cbar.set_ticks(np.linspace(0,2*10**(-6),5))
    # cbar.set_ticklabels(('0', '0.5', '1','1.5','2'))
    # cbar.set_ticks()#cbar_ticks
    cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=15)  # 主刻度
    cbar.ax.tick_params(which="minor", direction="in", length=0)  # 副刻度

    ##添加文本，保存图像
    if snrg[1] <= 120:
        plt.text(0.55, 5, f'SNRg=[{snrg[0]},{snrg[1]}]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\logg_{snrg[0]}_{snrg[1]}.png')
    else:
        plt.text(0.55, 5, f'SNRg=[{snrg[0]},+∞]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\logg_{snrg[0]}_+∞.png')
    plt.close()

def plot_scatter_feh(x,y,snrg):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    z = (z - min(z)) / (max(z) - min(z))
    sevp_i = x   # 观测数据
    estimate_i = y # 预测数据


    # ----------- Define Parameters ------------
    # radius = 3  # 半径
    colormap = plt.get_cmap("jet")  # 色带
    marker_size = 3  # 散点大小
    xrange = [-2.5, 1]
    yrange = [-2.5, 1]
    xticks = np.linspace(-2.5, 1, 3)
    yticks = np.linspace(-2.5, 1, 3)
    xlabel = "Feh_LAMOST / dex"
    ylabel_i = "Feh_FC / dex"
    ylabel_ii = "Estimate-2"
    ylabel_iii = "Estimate-3"
    # cbar_ticks = [10**0, 10**1, 10**2, 10**3, 10**4, 10**5]
    font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 15}
    ###################################plot
    plt.figure()
    plt.subplot(1, 1, 1, aspect="equal")
    plt.scatter(sevp_i, estimate_i, c=z, cmap=colormap, marker=".", s=marker_size)#norm=colors.LogNorm(vmin=z.min(), vmax=0.5 * z.max())
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
    cbar = plt.colorbar()  # 显示色带orientation='horizontal',extend="both", pad=0.1
    # cbar.set_label("Scatter Density", fontdict=font)
    # cbar.set_ticks(np.linspace(0,2*10**(-6),5))
    # cbar.set_ticklabels(('0', '0.5', '1','1.5','2'))
    # cbar.set_ticks()#cbar_ticks
    cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=15)  # 主刻度
    cbar.ax.tick_params(which="minor", direction="in", length=0)  # 副刻度

    ##添加文本，保存图像
    if snrg[1] <= 120:
        plt.text(-2.3, 0.65, f'SNRg=[{snrg[0]},{snrg[1]}]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\feh_{snrg[0]}_{snrg[1]}.png')
    else:
        plt.text(-2.3, 0.65, f'SNRg=[{snrg[0]},+∞]', fontsize=15)
        plt.savefig(f'E:\项目\参数测量\实验一\同一\图像\\feh_{snrg[0]}_+∞.png')
    plt.close()

data_teff = pd.read_csv('E:\项目\参数测量\实验一\同一/dr6_50w_snrg_teff.csv')
data_logg = pd.read_csv('E:\项目\参数测量\实验一\同一/dr6_50w_snrg_logg.csv')
data_feh = pd.read_csv('E:\项目\参数测量\实验一\同一/dr6_50w_snrg_feh.csv')
# data = data.iloc[0:20000]
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

    ##做Herotus_LAMOST
    x = teff['teff_dr6_test']  # 观测数据
    y = logg['logg_dr6_test']  # 预测数据
    x = np.asarray(x).T
    y = np.asarray(y).T
    plot_scatter_Herotus_LAMOST(x, y,snrg_choice[i])

    ##做Herotus_FC
    x = teff['teff_pre']  # 观测数据
    y = logg['logg_pre']  # 预测数据
    x = np.asarray(x).T
    y = np.asarray(y).T
    plot_scatter_Herotus_FC(x, y, snrg_choice[i])

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
