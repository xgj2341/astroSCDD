###数据预处理
## 1. csv2pkl_pa(): 将参数的csv文件整理保存为pkl文件
## 2. jiaya():解压压缩文件，删除压缩文件
## 3. datanormal():
#       3.1 删除坏像素点
#       3.2 补齐波长
#       3.3 光谱数据标准化

def csv2pkl_pa(path):
    import csv
    import pandas as pd
    f = open(path+'pa_7_5000.csv')
    csvreader = csv.reader(f)
    data = []
    obsid = []
    lmjd = []
    planid = []
    spid = []
    fiberid = []
    snrgs = []
    subclass = []
    teffs = []
    loggs = []
    fehs = []
    rvs = []
    for i, row in enumerate(csvreader):
        if i == 0:
            continue
        row = row[0].split('|')
        obsid.append(row[0])
        lmjd.append(row[2])
        planid.append(row[4])
        spid.append(row[5])
        fiberid.append(row[6])
        snrgs.append(row[7])
        subclass.append(row[9])
        teffs.append(row[13])
        loggs.append(row[14])
        fehs.append(row[15])
        rvs.append(row[16])

    df = pd.DataFrame()
    df['obsid'] = obsid
    df['lmjd'] = lmjd
    df['planid'] = planid
    df['spid'] = spid
    df['fiberid'] = fiberid
    df['snrgs'] = snrgs
    df['subclass'] = subclass
    df['teffs'] = teffs
    df['loggs'] = loggs
    df['fehs'] = fehs
    df['rvs'] = rvs
    df.to_pickle(path + 'pa_7_5000.pickl')

def jieya(filePath):
    import gzip
    import os
    def un_gz(file_name):
        f_name = file_name.replace(".gz", "")
        g_file = gzip.GzipFile(file_name)
        open(f_name, "wb+").write(g_file.read())
        g_file.close()

    filesnames = os.listdir(filePath)
    i = 1
    for filesname in filesnames:
        un_gz(filePath + filesname)
        os.remove(filePath + filesname)
        if i % 1000 == 0:
            print('已解压：', i)
        i += 1

def data_add(wave, flux, wave_temp):
    import sys
    import numpy as np
    sys.path.append(r'E:\项目\参数测量\astroslam-master\astroslam-master\slam')
    import normalization
    flux_norm, flux_cont = normalization.normalize_spectrum(wave, flux, (4000., 8000.), 100., p=(1E-8, 1E-7),
                                                            q=0.5, rsv_frac=2.0)
    ##去掉坏像素点
    flux_rmse = np.std(flux - flux_cont)
    index = np.where(flux - flux_cont > flux_rmse * 1.5)
    flux_good = flux
    flux_good[index] = flux_cont[index]

    ##补齐3909维数组
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
    import pyfits
    import numpy as np
    import sys
    sys.path.append(r'E:\项目\参数测量\astroslam-master\astroslam-master\slam')
    import normalization
    import warnings
    warnings.filterwarnings("ignore")

    pa = pd.read_pickle('H:\DR6_all\数据说明\pa_7_5000.pickl')
    wave_temp = pd.read_pickle('E:\项目\参数测量\数据\光谱数据\wave_temp.pkl')
    wave_anchor = wave_temp['wave_temp'].values

    fluxs_norm = []
    for i in range(len(pa)):
        filename = 'spec-' + pa['lmjd'][i] + '-' + pa['planid'][i]  +'_sp'+ pa['spid'][i].zfill(2)+'-'+ pa['fiberid'][i].zfill(3) + '.fits'
        data = pyfits.open('H:\DR6_all\DR6_7_5000\\' + filename)[0].data_train
        flux = data_add(wave= data[2],flux=data[0],wave_temp=wave_anchor)
        flux_norm, flux_cont = normalization.normalize_spectrum(wave_anchor, flux, (4000., 8000.), 100., p=(1E-8, 1E-7),
                                                                q=0.5, rsv_frac=2.0)
        fluxs_norm.append(flux_norm)
        if i%100==0:
            print('已完成：',i, '条')
    pa['fluxs_norm'] = fluxs_norm
    pa.to_pickle('H:\DR6_all\DR6_pickle\\7_5000.pkl')


