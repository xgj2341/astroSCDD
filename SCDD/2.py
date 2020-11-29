import pickle
import numpy as np
import catboost as cbt
import pandas as pd

def evaluation_index(real_values, pre_values):
    MD = np.mean(np.abs(real_values - pre_values))
    MSE = np.mean((real_values - pre_values) ** 2)
    RMSE = MSE ** (0.5)
    return MD, MSE, RMSE

path = 'H:\DR6_all\代码\实验一\data\\features\\'
data_train = pd.read_pickle(path + 'data_train.pkl')
data_test = pd.read_pickle(path + 'data_test.pkl')

obsid_dr6_train = data_train['obsid_dr6'].values
snrg_train = np.asarray(list(map(float,data_train['snrgs_dr6'])))
features_train = data_train.iloc[:,6:]
features_train = np.array(features_train.values.tolist()).reshape(-1,341)
subclasses_train = data_train['subclass']
teffs_train = np.asarray(list(map(float,data_train['teffs_dr6'])))
fehs_train = np.asarray(list(map(float,data_train['fehs_dr6'])))
loggs_train = np.asarray(list(map(float,data_train['loggs_dr6'])))

obsid_dr6_test = data_test['obsid_dr6'].values
snrg_test = np.asarray(list(map(float,data_test['snrgs_dr6'])))
features_test = data_test.iloc[:,6:]
features_test = np.array(features_test.values.tolist()).reshape(-1,341)
subclasses_test = data_test['subclass']
teffs_test = np.asarray(list(map(float,data_test['teffs_dr6'])))
fehs_test = np.asarray(list(map(float,data_test['fehs_dr6'])))
loggs_test = np.asarray(list(map(float,data_test['loggs_dr6'])))

###选择teff, logg, feh
x_train = features_train
x_test = features_test
y_train = loggs_train
y_test = loggs_test
# initialize
train_pool = cbt.Pool(x_train, y_train)
test_pool = cbt.Pool(x_test,y_test)

# specify the training parameters
model = cbt.CatBoostRegressor(iterations=500000,
                              learning_rate=0.01,
                              use_best_model=True,
                              early_stopping_rounds=800,
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
pickle.dump(model, open( 'H:\DR6_all\代码\实验一\model\\dr6_50w_snrg_' + 'logg2' +'.pickle.dat', "wb"))
# some time later...
# load model from file
loaded_model = pickle.load(open('H:\DR6_all\代码\实验一\model\\dr6_50w_snrg_' + 'logg2' +'.pickle.dat', "rb"))
y_preds = loaded_model.predict(x_test)

MD, MSE,RMSE = evaluation_index(y_test, y_preds)
print('teff平均绝对误差: ', MD,
      'teff均方误差：', MSE,
      'teff均方根误差：', RMSE)


f = open('H:\DR6_all\代码\实验一\\result\\dr6_50w_snrg_' + 'logg2' +'.csv', 'w', encoding='utf-8', newline="")
import csv
csv_writer = csv.writer(f)
csv_writer.writerow(['obsid_dr6','logg_dr6_test','logg_pre','snrg_test','subclass'])
for i in range(len(y_preds)):
    csv_writer.writerow([obsid_dr6_test[i], y_test[i], y_preds[i],snrg_test[i],subclasses_test.iloc[i]])
f.close()
###画图###########################################################################


