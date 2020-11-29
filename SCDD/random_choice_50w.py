def random_choice_50w():
    import pyfits
    import numpy as np
    import pandas as pd
    hud_pa = pyfits.open('.\\data\\数据说明\\dr6_v1_stellar_LSR.fits')[1].data_train
    num = len(hud_pa.snrg)
    arr = np.arange(num)
    np.random.shuffle(arr)
    hud_pa = hud_pa[arr[0:500000]]
    obsid_dr6 = hud_pa.obsid
    subclass = hud_pa.subclass
    teffs_dr6 = hud_pa.teff
    loggs_dr6 = hud_pa.logg
    fehs_dr6 = hud_pa.feh
    snrgs_dr6 = hud_pa.snrg
    lmjd = hud_pa.lmjd
    planid = hud_pa.planid
    spid = hud_pa.spid
    fiberid = hud_pa.fiberid

    df= pd.DataFrame({'obsid_dr6': obsid_dr6,'subclass': subclass,
                      'teffs_apogee': teffs_dr6, 'loggs_apogee': loggs_dr6,'fehs_apogee': fehs_dr6,
                      'snrgs_dr6': snrgs_dr6,
                      'lmjd':lmjd,'planid':planid,'spid':spid,'fiberid':fiberid})
    df.to_pickle(f'.\\data\\数据说明\dr6_v1_stellar_LSR_500000.pkl')




