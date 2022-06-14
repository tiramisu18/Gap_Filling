import numpy as np
import h5py
import time
from osgeo import gdal
import numpy.ma as ma
import ReadDirFiles
import Public_Methods

# 传入HDF文件的QC层信息，将十进制转为8位的二进制数据
def read_QC(QC, url):
    QC_Bin = []
    for idx in range(0, 46):
        print(idx)
        file_bin = []
        for i in range(0, 2400):
            row_bin = []
            for j in range(0, 2400):
                one = np.binary_repr(QC[idx][i][j], width=8) #将10进制转换为2进制
                row_bin.append(one)
            file_bin.append(row_bin)
        QC_Bin.append(file_bin)
    # return QC_Bin
    np.save(url, QC_Bin)

# 读取二进制QC(bin)的算法路径，转为相应的权重
def QC_AgloPath(QCBin, url):
    QC_Wei = []
    for idx in range(0, 46):
        print(idx)
        file_wei = []
        for i in range(0, 2400):
            weight = 0 
            row_wei = []
            for j in range(0, 2400):
                one = str(QCBin[idx][i][j])[0:3]
                if one == '000' or  one == '001': weight = 10
                elif one == '010' or one == '011' : weight = 5
                else: weight = 0
                row_wei.append(weight)
            file_wei.append(row_wei)
        QC_Wei.append(file_wei)
    np.save(url, QC_Wei)

# 读取二进制QC的云信息，转为相应的权重
def QC_CloudState(QCBin):
    QC_Wei = []
    for idx in range(0, 46):
        print(idx)
        file_wei = []
        for i in range(0, 2400):
            weight = 0 
            row_wei = []
            for j in range(0, 2400):
                one = str(QCBin[idx][i][j])[3:5]
                if one == '00' : weight = 10
                elif one == '01' : weight = 6
                elif one == '10' : weight = 3
                else: weight = 0
                row_wei.append(weight)
            file_wei.append(row_wei)
        QC_Wei.append(file_wei)
    np.save('../QC/h11v04_2018_CloudState_Wei', QC_Wei)

# 读取mat存储的MQC
def read_MQC (path, savepath):
    MQC_File=h5py.File(path) 
    # print(MQC_File.keys())
    file_Content = MQC_File["MQC_Score"]
    print('begin', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    MQC_All = []
    for idx in range(0, 44):
        MQC_data = MQC_File[file_Content[0,idx]]  # [column, row]
        MQC_Score = np.transpose(MQC_data[:])
        MQC_All.append(MQC_Score)
    print(len(MQC_All))
    print('end', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    np.save(savepath, MQC_All)

# 读取HDF文件数据集
def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    StdLAI = gdal.Open(subdatasets[5][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC, 'StdLAI': StdLAI}


tile = 'h12v04'
fileLists = ReadDirFiles.readDir('../HDF/%s' % tile)

# # print(fileLists[15])
LAIDatas, QCDatas, StdLAIDatas = [], [], []
# QCDatas = []
# StdLAIDatas = []
for file in fileLists:
    result = ReadFile(file)
    LAIDatas.append(result['LAI'])
    QCDatas.append(result['QC'])
    StdLAIDatas.append(result['StdLAI'])
raw_LAI = np.array(LAIDatas, dtype=float)

# # 将QC转为对应的权重
# # read_QC(QCDatas, '../QC/Version_1/%s_2018/%s_Bin' % (tile, tile))
# # QC_bin = np.load('../QC/Version_1/%s_2018/%s_Bin.npy' % (tile, tile))
# # QC_weight = QC_AgloPath(QC_bin, '../QC/Version_1/%s_2018/%s_AgloPath_Wei' % (tile, tile))

# 加上StdLAI作为权重的一部分
qualityControl = np.load('../QC/Version_1/%s_2018/%s_AgloPath_Wei.npy' % (tile, tile))
Public_Methods.render_Img(qualityControl[15], issave=True, savepath='./Daily_cache/0614/0614_1')
pos = qualityControl == 5
qualityControl[pos] = 3
Public_Methods.render_Img(ma.masked_equal(qualityControl[15]/10, 0), title = '', issave=True,savepath='./Daily_cache/0614/0614_2')

std = ma.masked_greater(StdLAIDatas, 100)
map_std = 1 + ((0.3 - 1) / (0.5 + ma.max(std) - ma.min(std))) * (std - ma.min(std)) # 数据归一化映射
Public_Methods.render_Img(std[15], issave=True, savepath='./Daily_cache/0614/0614_3')
aa = np.array(ma.filled(map_std, 1))

final_qualityControl = aa * qualityControl
Public_Methods.render_Img(ma.masked_equal(final_qualityControl[15]/10, 0), title='', issave=True,savepath='./Daily_cache/0614/0614_4')

# np.save('../QC/Version_3/%s_Weight' % (tile), final_qualityControl)
# np.save('../QC/Version_2/%s_2018/%s_Weight' % (tile, tile), final_qualityControl)


# 按照植被类型计算像元平均年季曲线变化
def landCover_Improved(raw, spatial, temporal, landCover, lcType):    
    raw_mean = []
    spa_mean= []
    tem_mean = []
    for i in range(0, 46):
        raw_ma = ma.array((ma.masked_greater(raw[i], 70)), mask=(landCover != lcType))
        spa_ma = ma.array((ma.masked_greater(spatial[i], 70)), mask=(landCover != lcType))
        tem_ma = ma.array((ma.masked_greater(temporal[i], 70)), mask=(landCover != lcType))
        raw_mean.append(np.mean(raw_ma)/10)
        spa_mean.append(np.mean(spa_ma)/10)
        tem_mean.append(np.mean(tem_ma)/10)

    Public_Methods.draw_polt_Line(np.arange(0, 361, 8),{
        'title': 'B%s'% lcType ,
        'xlable': 'Day',
        'ylable': 'LAI',
        'line': [raw_mean, spa_mean, tem_mean],
        'le_name': ['Tem3','Tem2', 'Temporal'],
        'color': ['gray', '#bfdb39', '#fd7400'],
        'marker': [',', 'o', '^', '.' ],
        'size': {'width': 10, 'height': 6},
        'lineStyle': ['dashed']
        },'./Daily_cache/0614/LC/lc_part_%s'% lcType, True, 1)

LC_file = gdal.Open('../LC/MCD12Q1.A2018001.h12v04.006.2019199202045.hdf')
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()

tem = []
tem2 = []
tem3 = []
for i in range(1, 47):
    print(i)
    tem2_data = np.load('./Daily_cache/0614/Temporal/LAI_%s.npy' % (i))
    tem3_data = np.load('./Daily_cache/0614/Temporal1/LAI_%s.npy' % (i))
    tem_data = np.load('../Improved_RealData/%s_2018/Temporal/LAI_%s.npy' % (tile, i))
    tem.append(tem_data)
    tem2.append(tem2_data)
    tem3.append(tem3_data)
tem_LAI = np.array(tem)
tem2_LAI = np.array(tem2)
tem3_LAI = np.array(tem3)
lcType = 6
landCover_Improved(tem3_LAI, tem2_LAI, tem_LAI, landCover, lcType)


