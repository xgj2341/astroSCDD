def download_sptra():
    import wget
    path = './/data//data//'
    f = open('.\数据描述\\url_LSR.txt', 'r')
    lines = f.readlines()
    for i, line in enumerate(lines):
        file_name = wget.download(line, path)
        print(str(i))