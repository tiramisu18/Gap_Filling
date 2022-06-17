import pandas as pd


# 将站点按照h、v所在位置分类
def classification_site():
    # data1 = pd.read_csv('数据筛选.csv', usecols=['Site name', 'DOY'], nrows=5)
    data1 = pd.read_csv('数据筛选.csv', usecols= ['Satellite', 'Site name', 'DOY', 'Site value', 'h', 'v', 'line', 'samp', 'c6 DOY'])
    # print(data1)
    h = 8
    v = 5
    specific = data1.loc[(data1['h'] == h) & (data1['v'] == v)] # 选定特定条件下的数据
    # specific = data1.loc[(data1['Site name'] == 'MOAB') ]
    count = len(specific)
    print(count)

    
    specific.to_csv('./Site_Classification/站点_h%02dv%02d.csv' % (h, v))

classification_site()


