import pandas as pd
import numpy as np
path = 'J:\DR6_all\代码\实验一\data\\features\\all_features\\'
datas = pd.DataFrame()
for i in range(10):
    for j in range(5):
        data = pd.read_pickle(path + 'dr6_50w_features_' + str(i) + '_' + str(j + 1) + '.pkl')
        del data['fluxs']
        del data['waves']
        del data['fluxs_norm']
        datas = pd.concat([datas,data])
        print(str(i) + '---'+str(j))

num_example = len(datas)
arr = np.arange(num_example)
np.random.shuffle(arr)
datas = datas.iloc[arr]


train_size = 0.8
train = datas[0:int(num_example*0.8)]
train.to_pickle('J:\DR6_all\代码\实验一\data\\features\\all_features\\train.pkl')
test = datas[int(num_example * 0.8):num_example]
test.to_pickle('J:\DR6_all\代码\实验一\data\\features\\all_features\\test.pkl')
s = 1