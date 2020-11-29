import catboost as cbt
import pickle
import pandas as pd
import time

x_test = pd.read_pickle('J:\DR6_all\代码\实验一\data\\features\\data_test.pkl').iloc[:,6:]
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

loaded_model = pickle.load(open(f'J:\DR6_all\代码\实验一\model\\dr6_50w_snrg10_teff.pickle.dat', "rb"))
start = time.time()
y_preds = loaded_model.predict(x_test)
end = time.time()
print(end-start)
s = 1