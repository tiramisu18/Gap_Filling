import os
import csv
import Read_Tiff

def readDir(dirPath):
    if dirPath[-1] == '/':
        print('path can not end with /')
        return
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        fileList.sort()
        # print(fileList[17])
        for f in fileList:
            f = dirPath+'/'+f
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles = subFiles + allFiles #合并当前目录与子目录的所有文件路径
            else:
                if f.find('TXT') != -1: allFiles.append(f)
        print('allFiles', len(allFiles))
        return allFiles
    else:
        return 'error, not a dir'
# 判断是否闰年
def f(n):
    if n % 4 == 0 and n % 100 != 0 or n % 400 == 0:
        return True
    else:
        return False

# 日期转天数
def dateToDay(date):
    y = int(date[:4])
    m = int(date[4:6])
    d = int(date[6:])
    month = [0, 31, 0, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if f(y):
        month[2] = 29
    else:
        month[2] = 28
    sum = 0
    for i in range(1, m):
        sum += month[i]
    sum += d

    return '%d%03d' % (y,sum)
    # print("2018%03d" % sum)

def filter_ValidPixels():
    fileLists = readDir('./Raw_Data/LP03')
    sites = {
        'BART': {'h': 12, 'v': 4, 'line': 1424.16, 'samp': 2105.61},
        'BLAN': {'h': 11, 'v': 5, 'line': 225.04, 'samp': 2250.38},
        'CPER': {'h': 10, 'v': 4, 'line': 2203.77, 'samp': 173.89},
        'DSNY': {'h': 10, 'v': 6, 'line': 449.49, 'samp': 1962.62},
        'DELA': {'h': 10, 'v': 5, 'line': 1789.49, 'samp': 1435.02},
        'GUAN': {'h': 11, 'v': 7, 'line': 486.81, 'samp': 1533.85},
        'HARV': {'h': 12, 'v': 4, 'line': 1790.43, 'samp': 1636.72},
        'JERC': {'h': 10, 'v': 5, 'line': 2112.74, 'samp': 1858.18},
        'JORN': {'h': 8, 'v': 5, 'line': 1777.73, 'samp': 2394.90},
        'KONA': {'h': 10, 'v': 5, 'line': 212.99, 'samp': 1207.90},
        'LAJA': {'h': 11, 'v': 7, 'line': 474.40, 'samp': 1490.80},
        'MOAB': {'h': 9, 'v':5, 'line': 419.89, 'samp': 981.96},
        'NIWO': {'h': 9, 'v':4, 'line': 2386.47, 'samp': 2203.54},
        'ONAQ': {'h': 9, 'v':4, 'line': 2356.88, 'samp': 978.91},
        'ORNL': {'h': 11, 'v':5, 'line': 968.11, 'samp': 427.40},
        'OSBS': {'h': 10, 'v':6, 'line': 77.22, 'samp': 2099.01},
        'SCBI': {'h': 11, 'v':5, 'line': 265.20, 'samp': 2203.28},
        'SERC': {'h': 12, 'v':5, 'line': 265.86, 'samp': 97.75},
        'STEI': {'h': 11, 'v':4, 'line': 1077.35, 'samp': 1731.83},
        'STER': {'h': 10, 'v':4, 'line': 2288.63, 'samp': 386.25},
        'SRER': {'h': 8, 'v':5, 'line': 1940.94, 'samp': 1419.03},
        'TALL': {'h': 10, 'v':5, 'line': 1691.39, 'samp': 1599.03},
        'UNDE': {'h': 11, 'v':4, 'line': 903.35, 'samp': 1935.23},
        'WOOD': {'h': 11, 'v':4, 'line': 688.72, 'samp': 594.74},
    }
    validedFiles = []
    for file in fileLists:
        with open(file, 'r') as fo:
            line = fo.read()
            line = line.splitlines()
            ValidPixels = line[27].split('=')
            # print ("读取的字符串: %s" % (ValidPixels))
            if float(ValidPixels[1]) > 50:
                fullName = fo.name.split('/')[-1]
                split_ = fullName.split('_')
                siteName = split_[3]
                doy = dateToDay(split_[4])
                tifUrl = f'{fo.name[:-10]}300M.TIF'
                siteValue = Read_Tiff.calculate_Mean(tifUrl)
                c6Doy = ((int(str(doy)[-3:]) - 1) // 8) * 8 + 1
                validedFiles.append([fullName, split_[2], siteName, split_[4], doy, siteValue, sites[f'{siteName}']['h'], sites[f'{siteName}']['v'],sites[f'{siteName}']['line'],sites[f'{siteName}']['samp'], '%s%03d'% (str(doy)[:4],c6Doy)])

   
    print(len(validedFiles))
    # print(validedFiles)
    # 写入到csv文件中
    # title = ['Valid pixels(%) > 50', 'Satellite', 'Site name', 'Date', 'DOY', 'Site value', 'h', 'v', 'line', 'samp', 'c6Doy']
    # with open("数据筛选.csv",'w',newline='') as file:#numline是来控制空的⾏数的
    #     writer=csv.writer(file)#这⼀步是创建⼀个csv的写⼊器
    #     writer.writerow(title)#写⼊标签
    #     writer.writerows(validedFiles)#写⼊样本数据

filter_ValidPixels()

# 读取csv
# with open("数据筛选.csv") as csvFile:
#     reader=csv.reader(csvFile) 
#     column=[row[0] for row in reader]
#     print(column)




# dateToDay('20180507')