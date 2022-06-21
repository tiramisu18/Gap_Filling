import numpy as np
import h5py
import time
from osgeo import gdal
import numpy.ma as ma
from pathlib import Path
import os
import ReadDirFiles
import Public_Methods

# 传入HDF文件的QC层信息，将十进制转为8位的二进制数据
def read_QC(QC, url, hv):    
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
    if Path(url).is_dir(): np.save(f'{url}/{hv}_Bin', QC_Bin)
    else:
        os.mkdir(url)
        np.save(f'{url}/{hv}_Bin', QC_Bin)

# 读取二进制QC(bin)的算法路径，转为相应的权重
def QC_AgloPath(hv, url):
    QC_bin = np.load(f'../QC/Version_1/{hv}_2018/{hv}_Bin.npy')
    QC_Wei = []
    for idx in range(0, 46):
        print(idx)
        file_wei = []
        for i in range(0, 2400):
            weight = 0 
            row_wei = []
            for j in range(0, 2400):
                one = str(QC_bin[idx][i][j])[0:3]
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

# 加上StdLAI作为权重的一部分
def addStdLAI(StdLAIDatas, hv, url):
    qualityControl = np.load(f'../QC/Version_1/{hv}_2018/{hv}_AgloPath_Wei.npy')
    # 将质量等级为5的备用算法修改为3
    pos = qualityControl == 5
    qualityControl[pos] = 3
    # StdLAI有效值范围为0-100
    std = ma.masked_greater(StdLAIDatas, 100)
    # 数据归一化映射
    map_std = 1 + ((0.3 - 1) / (0.5 + ma.max(std) - ma.min(std))) * (std - ma.min(std)) 
    surplus = np.array(ma.filled(map_std, 1))
    final_qualityControl = surplus * qualityControl
    if ~Path(url).is_dir():os.mkdir(url)
    np.save(f'{url}/{hv}_Weight', final_qualityControl)
    print(f'{hv} end')

# 读取HDF文件数据集
def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    StdLAI = gdal.Open(subdatasets[5][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC, 'StdLAI': StdLAI}


hv = 'h09v05'
fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')
LAIDatas, QCDatas, StdLAIDatas = [], [], []
for file in fileLists:
    result = ReadFile(file)
    LAIDatas.append(result['LAI'])
    QCDatas.append(result['QC'])
    StdLAIDatas.append(result['StdLAI'])
raw_LAI = np.array(LAIDatas, dtype=float)

# 将QC转为对应的权重
read_QC(QCDatas, f'../QC/Version_1/{hv}_2018', hv)
QC_AgloPath(hv, f'../QC/Version_1/{hv}_2018/{hv}_AgloPath_Wei')
addStdLAI(StdLAIDatas, hv, f'../QC/Version_2/{hv}_2018')


