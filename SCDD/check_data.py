##判断是否有需要用到的数据,如果是解压文件解压，否则不做处理
def check_data():
    import pandas as pd
    import os
    import gzip
    def un_gz(file_name):
        f_name = file_name.replace(".gz", "")
        g_file = gzip.GzipFile(file_name)
        open(f_name, "wb+").write(g_file.read())
        g_file.close()

    hud_pa = pd.read_pickle(f'.//数据描述\dr6_v1_stellar_LSR_500000.pkl')
    for i in range(500000):
        filesname_temp = 'spec-' + str(hud_pa['lmjd'][i]) + '-' + str(hud_pa['planid'][i], encoding="utf-8").strip() \
                         + '_sp' + str(hud_pa['spid'][i]).zfill(2) + '-' + str(hud_pa['fiberid'][i]).zfill(3) + '.fits'
        path = './/data\\data\\'

        if os.path.exists(path + filesname_temp + '.gz'):
            un_gz(path + filesname_temp + '.gz')
            os.remove(path + filesname_temp + '.gz')