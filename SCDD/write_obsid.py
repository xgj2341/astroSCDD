def write_obsid():
    import pandas as pd
    data = pd.read_pickle(f'.\\data\数据描述\dr6_v1_stellar_LSR_500000.pkl')
    obsid = data['obsid_dr6']
    f = open(f'.\\data\数据描述\dr6_v1_stellar_LSR_500000.txt','w')
    for row in obsid:
        f.write(str(row))
        f.write('\n')
    f.close()