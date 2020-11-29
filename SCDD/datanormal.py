from matplotlib import pyplot as plt
def data_add(wave, flux, wave_temp):
    import sys
    import numpy as np
    import normalization
    flux_norm, flux_cont = normalization.normalize_spectrum(wave, flux, (4000., 8000.), 100., p=(1E-8, 1E-7),
                                                            q=0.5, rsv_frac=2.0)
    ##
    flux_rmse = np.std(flux - flux_cont)
    index = np.where(flux - flux_cont > flux_rmse * 1.5)
    flux_good = flux
    flux_good[index] = flux_cont[index]

    ##
    diff_wave = np.abs(wave[0] - wave_temp)
    index_star = np.where(diff_wave==np.min(diff_wave))[0][0]
    diff_wave = np.abs(wave[-1] -wave_temp)
    index_stop = np.where(diff_wave==np.min(diff_wave))[0][0]
    flux_add = np.zeros(3909)
    flux_add[index_star:index_stop+1] = flux_good

    pa_left = np.polyfit(wave[0:50], flux_cont[0:50], 5)
    f_left = np.poly1d(pa_left)
    pa_right = np.polyfit(wave[-50:-1], flux_cont[-50:-1], 5)
    f_right = np.poly1d(pa_right)

    index = np.where(flux_add==0)[0]
    index_left = index[np.where(index<50)[0]]
    index_right = index[np.where(index >3859)[0]]

    if len(index_left)>0:
        flux_add[index_left] = f_left(wave_temp[index_left])
    if len(index_right)>0:
        flux_add[index_right] = f_right(wave_temp[index_right])
    return flux_add

def datanormal():
    import pandas as pd
    import normalization
    import warnings
    warnings.filterwarnings("ignore")

    for i in range(10):
        for j in range(5):
            filename = 'dr6_50w_' + str(i) + '_' + str(float(j+1)) + '.pkl'
            datas = pd.read_pickle('..\data\\fluxs//' + filename)
            datas = datas[datas['snrgs_dr6']>200]

            wave_temp = pd.read_pickle('..\\data\数据描述\wave_temp.pkl')
            wave_anchor = wave_temp['wave_temp'].values
            fluxs_norm = []

            for z in range(len(datas)):
                flux = data_add(wave= datas.iloc[z]['waves'],flux=datas.iloc[z]['fluxs'],wave_temp=wave_anchor)
                flux_norm, flux_cont = normalization.normalize_spectrum(wave_anchor, flux, (4000., 8000.), 100., p=(1E-8, 1E-7),
                                                                        q=0.5, rsv_frac=2.0)
                plt.plot(wave_anchor, flux_cont,zorder=3)
                plt.show()
                s = 1
                fluxs_norm.append(flux_norm)
                if z%100==0:
                    print('已完成：',z, '条')
            datas['fluxs_norm'] = fluxs_norm
            datas.to_pickle(f'\data\\normal\\dr6_50w_normal_{i}_{j+1}.pkl')

datanormal()