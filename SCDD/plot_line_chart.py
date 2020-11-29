import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel('E:\项目\参数测量\实验一\\result_all.xlsx')

fig = plt.figure()

#显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
ax1 = fig.add_subplot(111)

font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 15}

lns1 = ax1.plot(df['teff'], df['MD_teff'], 'r',label='MD',marker='o')
# ax1.set_ylabel('MD/K',fontdict={'weight': 'normal', 'size': 15})
# ax1.set_title("投资者用户数统计",fontdict={'weight': 'normal', 'size': 15})

lns2 = ax1.plot(df['teff'], df['RMSE_teff'],'g',label='RMSE',marker='o')
ax1.set_ylabel('MD,RMSE/K',fontdict=font)
plt.yticks( fontproperties='Times New Roman', size=12)
# ax1.set_title("投资者用户数统计",fontdict={'weight': 'normal', 'size': 15})
ax1.set_xlabel('Teff: S/N',fontdict={'family': 'Times New Roman',
            'weight': 'bold',
            'size': 14})

ax2 = ax1.twinx()  # this is the important function
lns3 = ax2.plot(df['teff'], df['MSE_teff'],'b',label='MSE',marker='o')
ax2.set_ylabel('MSE/$K^2$',fontdict=font)
plt.yticks( fontproperties='Times New Roman', size=12)
#参数rotation设置坐标旋转角度，参数fontdict设置字体和字体大小
ax1.set_xticklabels(df['teff'],rotation=0,fontdict={'family': 'Times New Roman',
            'weight': 'bold',
            'size': 13})

# added these three lines
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,fontsize=14)
fig.tight_layout()#调整整体空白
plt.show()
plt.savefig(f'E:\项目\参数测量\实验一\同一\line_chart\\teff.png')
plt.show()
s = 1