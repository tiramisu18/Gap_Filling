import pandas as pd


# data1 = pd.read_csv('数据筛选.csv', usecols=['Site name', 'DOY'], nrows=5)
data1 = pd.read_csv('数据筛选.csv', usecols= ['Satellite', 'Site name', 'DOY', 'Site value', 'h', 'v', 'line', 'samp', 'c6 DOY'])
# print(data1)
h = 12
v = 5
specific = data1.loc[(data1['h'] == h) & (data1['v'] == v)]
# specific = data1.loc[(data1['Site name'] == 'MOAB') ]
count = len(specific)
print(count)

# 按照tile分类
specific.to_csv(f'./Site_Classification/站点_h{h}v{v}_{count}.csv')
# 读取csv
# with open("数据筛选.csv") as csvFile:
#     reader=csv.reader(csvFile) 
#     column=[row[0] for row in reader]
#     print(column)

