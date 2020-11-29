import pickle
import numpy as np
import catboost as cbt
from catboost import cv
import pandas as pd
import time
import os

def evaluation_index(real_values, pre_values):
    MD = np.mean(np.abs(real_values - pre_values))
    MSE = np.mean((real_values - pre_values) ** 2)
    RMSE = MSE ** (0.5)
    return MD, MSE, RMSE
def chose_data(data):
    # snrg_dr6 = np.asarray(data['snrgs_dr6'].values)
    teffs_dr6 = np.asarray(data['teffs_dr6'].values)
    loggs_dr6 = np.asarray(data['loggs_dr6'].values)
    fehs_dr6 = np.asarray(data['fehs_dr6'].values)
    snrg = np.asarray(data['snrgs_dr6'].values)
    index = np.where((snrg>10) &
                     (teffs_dr6>4000) & (teffs_dr6<8000) &
                     (loggs_dr6>0.5) & (loggs_dr6<5.5) &
                     (fehs_dr6>-2.5) &(fehs_dr6<1.0))
    data = data.iloc[index]
    return data

path = 'J:\DR6_all\代码\实验一\data\\features\\all_features\\'
datas = pd.DataFrame()
for i in range(10):
    for j in range(5):
        star = time.time()
        data = pd.read_pickle(path + 'dr6_50w_features_' + str(i) + '_' + str(j + 1) + '.pkl')
        del data['fluxs'], data['waves'],data['fluxs_norm']
        datas = pd.concat([datas,data])
        print(str(i)+str(j))
teffs_train = np.asarray(list(map(float, datas[datas['chose']==1]['teffs_dr6'])))
teffs_test = np.asarray(list(map(float, datas[datas['chose']==2]['teffs_dr6'])))
column_name = list(datas)[6:-1]
minwindows = {}
for row in column_name:
    temp = row.split('_')
    if temp[1] in minwindows:
        minwindows[temp[1]] +=1
    else:
        minwindows[temp[1]] = 1

offset = 0
for row in minwindows:
    temp = minwindows[row]
    offset = offset + temp

    features_train = datas[datas['chose']==1].iloc[:,6:6 + offset ]
    features_test = datas[datas['chose']==2].iloc[:,6:6 + offset]
    features_train = np.array(features_train.values.tolist()).reshape(-1, offset * 1)
    features_test = np.array(features_test.values.tolist()).reshape(-1, offset * 1)
    # teffs = np.array(teffs.values.tolist())

    # 随机挑选
    # x_train, x_test, y_train, y_test = train_test_split(features, teffs,train_size=0.8, random_state=33)
    train_size = 0.8
    x_train = features_train
    x_test = features_test

    y_train = teffs_train
    y_test = teffs_test
    del features_train,features_test

    # initialize
    train_pool = cbt.Pool(x_train, y_train)
    test_pool = cbt.Pool(x_test,y_test)

    del x_train,x_test

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
    model.fit(train_pool, eval_set=test_pool, verbose=10)
    ##模型保存与恢复
    # save model to file
    pickle.dump(model, open(f'J:\DR6_all\代码\实验一\model\chose_windows\\{row}.pickle.dat', "wb"))
    os.rename('./catboost_info',f'./catboost_info_{row}')





# some time later...
# # load model from file
# loaded_model = pickle.load(open('J:\DR6_all\代码\实验一\model\chose_windows\\1.pickle.dat', "rb"))
# y_preds = loaded_model.predict(x_test)

# MD, MSE,RMSE = evaluation_index(y_test, y_preds)
# print('teff平均绝对误差: ', MD,
#       'teff均方误差：', MSE,
#       'teff均方根误差：', RMSE)
#
#
# f = open('H:\DR6_all\代码\实验一\model\\dr6_50w_snrg_' + str(snrg_l)+ '_'+ str(snrg_h) +'.csv', 'w', encoding='utf-8', newline="")
# import csv
# csv_writer = csv.writer(f)
# csv_writer.writerow(['obsid_dr6','teff_dr6_test','teff_pre','snrg_test'])
# for i in range(len(y_preds)):
#     csv_writer.writerow([obsid_dr6_test[i], y_test[i], y_preds[i],snrg_test[i]])
# csv_writer.writerow(['teff_MD',MD])
# csv_writer.writerow(['teff_MSE',MSE])
# csv_writer.writerow(['teff_RMSE',RMSE])
# f.close()

###画图###########################################################################


