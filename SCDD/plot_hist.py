from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_excel('H:\DR6_all\代码\实验一\\result\实验对比.xlsx')
# 显示高度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%.2f' % height, fontproperties='Times New Roman',fontsize=15)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure()
name_list = ['ours', 'SLAM','GSN','StarNet','Kernel']
num_list = [36.36238475,
50,
80,
100,
100
]
autolabel(plt.bar(range(len(num_list)), num_list, color='rgbmy', tick_label=name_list))
plt.xticks( fontproperties='Times New Roman', size=15)
plt.yticks( [0,30,60,90,120],fontproperties='Times New Roman', size=15)
plt.ylabel('RMSE Teff/K', fontproperties='Times New Roman',fontsize=15)
plt.xlabel('算法',fontsize=15)
plt.show()
plt.savefig(f'H:\DR6_all\代码\实验一\\result\同一\hist\\logg.png')
s = 1