import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import ReadDirFiles
import math
import h5py
import time
import random
import Filling_Pixel
import Public_Methods

def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC}

# 生成一个包含n个介于a和b之间随机整数的数组（不重复）
def int_random(a, b, n) :
    a_list = []
    while len(a_list) < n :
        d_int = random.randint(a, b)
        if(d_int not in a_list) :
            a_list.append(d_int)
        else :
            pass
    return a_list

def render_Img (data, title='Algo Path', issave=False, savepath=''):
    plt.imshow(data, cmap = plt.cm.jet)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    # plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()

def render_LAI (data, title='Image', issave=False, savepath=''):
    colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
    bounds = [0,10,20,30,40,50,60,70,250]
    cmap = pltcolor.ListedColormap(colors)
    norm = pltcolor.BoundaryNorm(bounds, cmap.N)
    plt.title(title, family='Times New Roman', fontsize=18)
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.axis('off')
    cbar = plt.colorbar()
    cbar.set_ticklabels(['0','1','2','3','4','5','6','7','250'])
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()

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


tile = 'h25v06'
fileLists = ReadDirFiles.readDir('../HDF/%s' % tile)

LAIDatas = []
QCDatas = []
for file in fileLists:
    result = ReadFile(file)
    LAIDatas.append(result['LAI'])
    QCDatas.append(result['QC'])

# 将QC转为对应的权重
# read_QC(QCDatas, '../QC/%s_2018/%s_Bin' % (tile, tile))
# QC_bin = np.load('../QC/%s_2018/%s_Bin.npy' % (tile, tile))
# QC_weight = QC_AgloPath(QC_bin, '../QC/%s_2018/%s_AgloPath_Wei' % (tile, tile))

LC_file = gdal.Open('../LC/MCD12Q1.A2018001.h25v06.006.2019200011117.hdf')
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()

qualityControl = np.load('../QC/%s_2018/%s_AgloPath_Wei.npy' % (tile, tile))

# 时空相关性计算
for index in range(0, 46): 
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(index)
    # Tem = Filling_Pixel.Temporal_Cal_Matrix_Tile(np.array(LAIDatas), index, landCover, qualityControl, 3,  0.5)
    # np.save('../Imporved_RealData/%s_2018/Temporal/LAI_%s'% (tile, (index + 1)), Tem)
    Spa = Filling_Pixel.Spatial_Cal_Matrix_Tile(np.array(LAIDatas), index, landCover, qualityControl, 2,  4)
    np.save('../Imporved_RealData/%s_2018/Spatial/LAI_%s'% (tile, (index + 1)), Spa)
