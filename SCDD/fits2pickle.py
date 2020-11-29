import pyfits
import numpy as np
import pandas as pd
import gzip
import os
def un_gz(file_name):
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()

def fits2pickle():
    fluxs = []
    hud_pa = pyfits.open('E:\DR6_all\\dr6_v1_stellar_LSR_snrg200.fits')
    row_num = len(hud_pa[1].data.obsid)
    obsid_dr6 = []
    subclass = []
    teffs_dr6 = []
    loggs_dr6 = []
    fehs_dr6 =  []
    snrgs_dr6_active = []
    # row_num = 200
    path = 'E:\DR6_all\DR6_140\\'
    # f = open('dr6_snrg140_add.txt','w')
    for i in range(row_num):

        filesname_temp = 'spec-' + str(hud_pa[1].data.lmjd[i]) + '-' + str(hud_pa[1].data.planid[i],encoding="utf-8").strip() \
                         + '_sp' + str(hud_pa[1].data.spid[i]).zfill(2) + '-' + str(hud_pa[1].data.fiberid[i]).zfill(3) + '.fits'
        if os.path.exists(path+filesname_temp +'.gz'):
            # un_gz(path+filesname_temp + '.gz')
            # os.remove(path+filesname_temp)
            hud = pyfits.open(path+filesname_temp)
            flux = np.zeros(3909)
            flux[0:len(hud[0].data[0])] = hud[0].data[0]
            fluxs.append(flux)
            obsid_dr6.append(hud_pa[1].data.obsid[i])
            subclass.append(hud_pa[1].data.subclass[i])
            teffs_dr6.append(hud_pa[1].data.teff[i])
            loggs_dr6.append(hud_pa[1].data.logg[i])
            fehs_dr6.append(hud_pa[1].data.feh[i])
            snrgs_dr6_active.append(hud_pa[1].data.snrg[i])
            # f.write(str(hud_pa[1].data.obsid[i]))
            # f.write('\n')
        if i%100==0:
            print('已完成： ', i ,'条')
    # f.close()

    obsid_dr6 = np.asarray(obsid_dr6)
    subclass = np.asarray(subclass)
    teffs_dr6 = np.asarray(teffs_dr6)
    loggs_dr6 = np.asarray(loggs_dr6)
    fehs_dr6 = np.asarray(fehs_dr6)
    snrgs_dr6_active = np.asarray(snrgs_dr6_active)
    fluxs = np.asarray(fluxs).reshape(-1,3909)

    snrgs_dr6_active = np.asarray(snrgs_dr6_active)

    snrg_200_index = np.where(snrgs_dr6_active > 200)
    snrg_250_index = np.where(snrgs_dr6_active > 250)
    snrg_300_index = np.where(snrgs_dr6_active > 300)


    snrg_200_250_index = list(set(snrg_200_index[0]) - set(snrg_250_index[0]))
    snrg_250_300_index = list(set(snrg_250_index[0]) - set(snrg_300_index[0]))
    snrg_300_index = list(snrg_300_index[0])

    # df_140_150 = pd.DataFrame({'obsid_dr6': obsid_dr6[snrg_140_150_index],
    #                    'subclass':subclass[snrg_140_150_index],
    #                    'teffs_apogee':teffs_dr6[snrg_140_150_index],'loggs_apogee':loggs_dr6[snrg_140_150_index], 'fehs_apogee':fehs_dr6[snrg_140_150_index],
    #                    'snrgs_dr6':snrgs_dr6_active[snrg_140_150_index],
    #                    'features':list(fluxs[snrg_140_150_index])})
    # df_140_150.to_pickle('E:\DR6_all\DR6_pickle\dr6_140_150.pkl')
    #
    # df_150_170 = pd.DataFrame({'obsid_dr6': obsid_dr6[snrg_150_170_index],
    #                    'subclass':subclass[snrg_150_170_index],
    #                    'teffs_apogee':teffs_dr6[snrg_150_170_index],'loggs_apogee':loggs_dr6[snrg_150_170_index], 'fehs_apogee':fehs_dr6[snrg_150_170_index],
    #                    'snrgs_dr6':snrgs_dr6_active[snrg_150_170_index],
    #                    'features':list(fluxs[snrg_150_170_index])})
    # df_150_170.to_pickle('E:\DR6_all\DR6_pickle\dr6_150_170.pkl')
    #
    # df_170_200 = pd.DataFrame({'obsid_dr6': obsid_dr6[snrg_170_200_index],
    #                    'subclass':subclass[snrg_170_200_index],
    #                    'teffs_apogee':teffs_dr6[snrg_170_200_index],'loggs_apogee':loggs_dr6[snrg_170_200_index], 'fehs_apogee':fehs_dr6[snrg_170_200_index],
    #                    'snrgs_dr6':snrgs_dr6_active[snrg_170_200_index],
    #                    'features':list(fluxs[snrg_170_200_index])})
    # df_170_200.to_pickle('E:\DR6_all\DR6_pickle\dr6_170_200.pkl')

    df_200_250 = pd.DataFrame({'obsid_dr6': obsid_dr6[snrg_200_250_index],
                       'subclass':subclass[snrg_200_250_index],
                       'teffs_apogee':teffs_dr6[snrg_200_250_index],'loggs_apogee':loggs_dr6[snrg_200_250_index], 'fehs_apogee':fehs_dr6[snrg_200_250_index],
                       'snrgs_dr6':snrgs_dr6_active[snrg_200_250_index],
                       'features':list(fluxs[snrg_200_250_index])})
    df_200_250.to_pickle('E:\DR6_all\DR6_pickle\dr6_200_250.pkl')

    df_250_300 = pd.DataFrame({'obsid_dr6': obsid_dr6[snrg_250_300_index],
                       'subclass':subclass[snrg_250_300_index],
                       'teffs_apogee':teffs_dr6[snrg_250_300_index],'loggs_apogee':loggs_dr6[snrg_250_300_index], 'fehs_apogee':fehs_dr6[snrg_250_300_index],
                       'snrgs_dr6':snrgs_dr6_active[snrg_250_300_index],
                       'features':list(fluxs[snrg_250_300_index])})
    df_250_300.to_pickle('E:\DR6_all\DR6_pickle\dr6_250_300.pkl')

    df_300 = pd.DataFrame({'obsid_dr6': obsid_dr6[snrg_300_index],
                       'subclass':subclass[snrg_300_index],
                       'teffs_apogee':teffs_dr6[snrg_300_index],'loggs_apogee':loggs_dr6[snrg_300_index], 'fehs_apogee':fehs_dr6[snrg_300_index],
                       'snrgs_dr6':snrgs_dr6_active[snrg_300_index],
                       'features':list(fluxs[snrg_300_index])})
    df_300.to_pickle('E:\DR6_all\DR6_pickle\dr6_300.pkl')

def fits2pickle_dr6AFGK_APOGEE():
    fluxs = []
    waves = []
    hud_pa = pyfits.open('E:\DR6_all\数据说明\dr6AFGK_apogee')
    row_num = len(hud_pa[1].data.obsid)
    obsid_dr6 = []
    subclass = []
    teffs_dr6 = []
    loggs_dr6 = []
    fehs_dr6 = []
    snrgs_dr6 = []

    obsid_apogee = []
    teffs_apogee = []
    loggs_apogee = []
    fehs_apogee = []
    snrs_apogee = []

    # row_num = 200
    path = 'E:\DR6_all\DR6_APOGEE\dr6_apogee\\'
    for i in range(row_num):

        filesname_temp = 'spec-' + str(hud_pa[1].data.lmjd[i]) + '-' + str(hud_pa[1].data.planid[i],encoding="utf-8").strip() \
                         + '_sp' + str(hud_pa[1].data.spid[i]).zfill(2) + '-' + str(hud_pa[1].data.fiberid[i]).zfill(3) + '.fits'

        hud = pyfits.open(path+filesname_temp)
        flux = hud[0].data[0]
        wave = hud[0].data[2]

        fluxs.append(flux)
        waves.append(wave)
        obsid_dr6.append(hud_pa[1].data.obsid[i])
        subclass.append(hud_pa[1].data.subclass[i])
        teffs_dr6.append(hud_pa[1].data.teff_1[i])
        loggs_dr6.append(hud_pa[1].data.logg_1[i])
        fehs_dr6.append(hud_pa[1].data.feh[i])
        snrgs_dr6.append(hud_pa[1].data.snrg[i])
        obsid_apogee.append(hud_pa[1].data.APOGEE_ID[i])
        teffs_apogee.append(hud_pa[1].data.TEFF_2[i])
        loggs_apogee.append(hud_pa[1].data.LOGG_2[i])
        fehs_apogee.append(hud_pa[1].data.FE_H[i])
        snrs_apogee.append(hud_pa[1].data.SNR[i])



        if i%100==0:
            print('已完成： ', i ,'条')

    obsid_dr6 = np.asarray(obsid_dr6)
    subclass = np.asarray(subclass)
    teffs_dr6 = np.asarray(teffs_dr6)
    loggs_dr6 = np.asarray(loggs_dr6)
    fehs_dr6 = np.asarray(fehs_dr6)
    snrgs_dr6 = np.asarray(snrgs_dr6)
    fluxs = np.asarray(fluxs)
    waves = np.asarray(waves)

    obsid_apogee = np.asarray(obsid_apogee)
    teffs_apogee = np.asarray(teffs_apogee)
    loggs_apogee = np.asarray(loggs_apogee)
    fehs_apogee = np.asarray(fehs_apogee)
    snrs_apogee = np.asarray(snrs_apogee)

    df = pd.DataFrame({'obsid_dr6': obsid_dr6,'subclass':subclass,
                       'teffs_dr6':teffs_dr6,'loggs_dr6':loggs_dr6, 'fehs_dr6':fehs_dr6,'snrgs_dr6':snrgs_dr6,
                       'flux':list(fluxs),'wave':list(waves),
                       'obsid_apogee':obsid_apogee, 'teffs_apogee': teffs_apogee, 'loggs_apogee': loggs_apogee, 'fehs_apogee': fehs_apogee, 'snrgs_apogee': snrs_apogee,
                       })
    df.to_pickle('E:\DR6_all\DR6_APOGEE\dr6_apogee.pkl')

    s = 1

fits2pickle_dr6AFGK_APOGEE()

