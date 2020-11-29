import pickle
import numpy as np
import catboost as cbt
import pandas as pd

def evaluation_index(real_values, pre_values):
    MD = np.mean(np.abs(real_values - pre_values))
    MSE = np.mean((real_values - pre_values) ** 2)
    RMSE = MSE ** (0.5)
    return MD, MSE, RMSE

path = '..\data\\features\\'
datas = []
for i in range(10):
    for j in range(5):
        data = pd.read_pickle(path + 'dr6_50w_features_' + str(i) + '_' + str(j + 1) + '.pkl')
        del data['fluxs']
        del data['waves']
        del data['fluxs_norm']

        snrg_dr6 = np.asarray(data['snrgs_dr6'].values)
        teffs_dr6 = np.asarray(data['teffs_dr6'].values)
        loggs_dr6 = np.asarray(data['loggs_dr6'].values)
        fehs_dr6 = np.asarray(data['fehs_dr6'].values)
        index = np.where((snrg_dr6>10) &
                         (teffs_dr6>4000) & (teffs_dr6<8000) &
                         (loggs_dr6>0.5) & (loggs_dr6<5.5) &
                         (fehs_dr6>-2.5) &(fehs_dr6<1.0))
        data = data.iloc[index]
        datas.append(data)
datas = pd.concat(datas)
data = datas

obsid_dr6 = data['obsid_dr6'].values
snrg = np.asarray(list(map(float,data['snrgs_dr6'])))
teffs = np.asarray(list(map(float,data['teffs_dr6'])))
fehs = np.asarray(list(map(float,data['fehs_dr6'])))
loggs = np.asarray(list(map(float,data['loggs_dr6'])))
features = data.iloc[:,6:]
features = np.array(features.values.tolist()).reshape(-1,341)
# teffs = np.array(teffs.values.tolist())

teffmax = max(teffs)
teffmin = min(teffs)
print(teffmin, teffmax)
fehmax = max(fehs)
fehmin = min(fehs)
print(fehmin, fehmax)
loggmax = max(loggs)
loggmin = min(loggs)
print(loggmin, loggmax)

# 随机挑选
num_example = features.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
features = features[arr]
teffs = teffs[arr]
loggs = loggs[arr]
fehs = fehs[arr]
snrg = snrg[arr]
obsid_dr6 = obsid_dr6[arr]

# x_train, x_test, y_train, y_test = train_test_split(features, teffs,train_size=0.8, random_state=33)
train_size = 0.8
x_train = features[0:int(num_example*0.8)]
x_test = features[int(num_example*0.8):num_example]

y_train = fehs[0:int(num_example*0.8)]
y_test = fehs[int(num_example*0.8):num_example]

snrg_test = snrg[int(num_example*0.8):num_example]
obsid_dr6_test = obsid_dr6[int(num_example*0.8):num_example]
# subclass_test = subclass[int(num_example*0.8):num_example]


# initialize
train_pool = cbt.Pool(x_train, y_train)
test_pool = cbt.Pool(x_test,y_test)

# specify the training parameters
model = cbt.CatBoostRegressor(iterations=500000,
                              learning_rate=0.01,
                              use_best_model=True,
                              early_stopping_rounds=500,
                              random_seed=42,
                              task_type='CPU', #没有GPU可设为CPU
                              # devices='0:1',
                              eval_metric='RMSE',
                              loss_function='RMSE',
                              )
#train the model
model.fit(train_pool,eval_set=test_pool, verbose=10)

##模型保存与恢复
# save model to file
pickle.dump(model, open(r"H:\DR6_all\代码\实验一\model\\dr6_50w_snrg10_feh.pickle.dat", "wb"))
# some time later...
# load model from file
loaded_model = pickle.load(open(r"H:\DR6_all\代码\实验一\model\\dr6_50w_snrg10_feh.pickle.dat", "rb"))
y_preds = loaded_model.predict(x_test)

MD, MSE,RMSE = evaluation_index(y_test, y_preds)
print('teff平均绝对误差: ', MD,
      'teff均方误差：', MSE,
      'teff均方根误差：', RMSE)


f = open(r'H:\DR6_all\代码\实验一\\result/dr6_50w_snrg10_feh.csv', 'w', encoding='utf-8', newline="")
import csv
csv_writer = csv.writer(f)
csv_writer.writerow(['obsid_dr6','feh_dr6_test','feh_pre','snrg_test'])
for i in range(len(y_preds)):
    csv_writer.writerow([obsid_dr6_test[i], y_test[i], y_preds[i],snrg_test[i]])
csv_writer.writerow(['teff_MD',MD])
csv_writer.writerow(['teff_MSE',MSE])
csv_writer.writerow(['teff_RMSE',RMSE])
f.close()
###画图###########################################################################


