def get_allwindows_features(data):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm, tqdm_notebook, tnrange

    fetures_dim = np.size(data,1)
    window_sizes = []
    window_size = fetures_dim
    while window_size >= 20:
        window_sizes.append(window_size)
        window_size = int(window_size/2)

    data_ = pd.DataFrame()
    for window_size in window_sizes:
        # print("Window size is:", window_size)
        len_ = int(fetures_dim / window_size)
        for i in tnrange(len_):
            tmp = data[:, i * window_size:(i + 1) * window_size].copy()
            # wave_star = wave[i * window_size]
            # wave_stop = wave[(i + 1) * window_size-1]
            pd_tmp = pd.DataFrame(tmp)
            data_[str(i) + '_' + str(window_size) + '_mean'] = tmp.mean(axis=1)
            data_[str(i) + '_' + str(window_size) + '_max'] = tmp.max(axis=1)
            data_[str(i) + '_' + str(window_size) + '_min'] = tmp.min(axis=1)
            data_[str(i) + '_' + str(window_size) + '_var'] = tmp.var(axis=1)
            data_[str(i) + '_' + str(window_size) + '_median'] = np.median(tmp, axis=1)
            data_[str(i) + '_' + str(window_size) + '_sum'] = tmp.sum(axis=1)
            data_[str(i) + '_' + str(window_size) + '_skew'] = pd_tmp.skew(axis=1).values
            data_[str(i) + '_' + str(window_size) + '_kurt'] = pd_tmp.kurt(axis=1).values
            data_[str(i) + '_' + str(window_size) + '_range'] = data_[str(i) + '_' + str(window_size) + '_max'] - data_[
                str(i) + '_' + str(window_size) + '_min']
            data_[str(i) + '_' + str(window_size) + '_argmax'] = pd_tmp.idxmax(axis=1).values
            data_[str(i) + '_' + str(window_size) + '_argmin'] = pd_tmp.idxmin(axis=1).values
    return data_

import pandas as pd

import numpy as np
import time
path = '..\代码\实验一\data\\'
for i in range(10):
    for j in range(5):
        stra = time.time()
        data = pd.read_pickle(path + 'normal\\dr6_50w_normal_' + str(i) + '_' + str(j+1) + '.pkl')
        fluxs_norm = []
        for row in data['fluxs_norm']:
            fluxs_norm.append(row)
        fluxs_norm = np.reshape(fluxs_norm,(10000,3909))
        fluxs_features = get_allwindows_features(fluxs_norm)
        data = pd.merge(left=data,right=fluxs_features,left_index=True,right_index=True)
        data.to_pickle(path + 'features\\all_features\\dr6_50w_features_' + str(i) + '_' + str(j+1) + '.pkl')
        end = time.time()
        print(str(end-stra))




