import numpy as np
import h5py
import time
from osgeo import gdal
import numpy.ma as ma
from pathlib import Path
import os
import ReadDirFiles
import Public_Methods

# 读取HDF文件数据集
def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    StdLAI = gdal.Open(subdatasets[5][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC, 'StdLAI': StdLAI}

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
    if not Path(url).is_dir(): os.mkdir(url)
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

# 权重加上StdLAI和相对TSS
# def addStdLAITSS(StdLAIDatas, TSSValues, hv, url):
#     qualityControl = np.load(f'../QC/Version_1/{hv}_2018/{hv}_AgloPath_Wei.npy')
#     # 将质量等级为5的备用算法修改为3
#     pos = qualityControl == 5
#     qualityControl[pos] = 3
#     # StdLAI有效值范围为0-100
#     std = ma.masked_greater(StdLAIDatas, 100)
#     control = 0.5 * std + 0.5 * TSSValues
#     map_control = 0.5 + ((0.15 - 0.5) / (ma.max(control) - ma.min(control))) * (control - ma.min(control)) 
#     surplus = np.array(ma.filled(map_control, 1))
#     final_qualityControl = surplus * qualityControl
#     if not Path(url).is_dir(): os.mkdir(url)
#     np.save(f'{url}/{hv}_Weight', final_qualityControl)
#     print(f'{hv} end')


# 权重加上StdLAI和相对TSS
def addStdLAITSS(StdLAIDatas, TSSValues, hv, url):
    qualityControl = np.load(f'../QC/Version_1/{hv}_2018/{hv}_AgloPath_Wei.npy')
    # 将质量等级为5的备用算法修改为3
    pos = qualityControl == 5
    qualityControl[pos] = 3
    # StdLAI有效值范围为0-100
    std = ma.masked_greater(StdLAIDatas, 100)
    map_std = 0.5 + ((0.15 - 0.5) / (0.5 + ma.max(std) - ma.min(std))) * (std - ma.min(std)) 
    tss = ma.array(TSSValues, mask = pos)
    # tss中大于1的值直接设为0.15，小于1的值映射到0.16-0.5
    map_tss = 0.5 + ((0.16 - 0.5) / (1 - 0) * (tss - 0)) 
    p2 = map_tss < 0
    map_tss[p2] = 0.15
    control = map_std + map_tss
    surplus = np.array(ma.filled(control, 1))
    final_qualityControl = surplus * qualityControl
    if not Path(url).is_dir(): os.mkdir(url)
    np.save(f'{url}/{hv}_Weight', final_qualityControl)
    print(f'{hv} end')


def cal_TSS(LAIDatas, index):
    numerators = np.absolute(((LAIDatas[index + 1] - LAIDatas[index - 1]) * index) - (LAIDatas[index] * 2) - ((LAIDatas[index + 1] - LAIDatas[index - 1]) * (index - 1)) + (LAIDatas[index - 1] * 2))
    denominators = np.sqrt(np.square(LAIDatas[index + 1] - LAIDatas[index - 1]) + 2**2)
    # absoluteTSS = (numerators / denominators) / 10
    absoluteTSS = numerators / denominators
    relativeTSS = np.round(absoluteTSS / LAIDatas[index], 2)
    return np.nan_to_num(relativeTSS, posinf=10, neginf=10)
    
# hv = 'h12v05'
hvLists = ['h08v05', 'h09v04', 'h09v05', 'h10v04', 'h10v05', 'h10v06', 'h11v04', 'h11v05', 'h11v07', 'h12v04', 'h12v05']


for hv in hvLists:
    print(hv)
    fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')
    LAIDatas, QCDatas, StdLAIDatas = [], [], []
    for file in fileLists:
        result = ReadFile(file)
        LAIDatas.append(result['LAI'])
        QCDatas.append(result['QC'])
        StdLAIDatas.append(result['StdLAI'])
    raw_LAI = np.array(LAIDatas)

    TSSArray = np.ones((1,raw_LAI.shape[1], raw_LAI.shape[2])) 
    for index in range(1,45):
        one = cal_TSS(raw_LAI, index)
        TSSArray = np.append(TSSArray, one.reshape(1, one.shape[0], one.shape[1]), axis=0)
    TSSArray = np.append(TSSArray, np.ones((1,raw_LAI.shape[1], raw_LAI.shape[2])), axis=0)

    
    # # 将QC转为对应的权重
    # read_QC(QCDatas, f'../QC/Version_1/{hv}_2018', hv)
    # QC_AgloPath(hv, f'../QC/Version_1/{hv}_2018/{hv}_AgloPath_Wei')
    addStdLAITSS(StdLAIDatas, TSSArray, hv, f'../QC/Version_4/{hv}_2018')


hv = 'h12v04'
fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')
LAIDatas, QCDatas, StdLAIDatas = [], [], []
for file in fileLists:
    result = ReadFile(file)
    LAIDatas.append(result['LAI'])
    # QCDatas.append(result['QC'])
    # StdLAIDatas.append(result['StdLAI'])
raw_LAI = np.array(LAIDatas)

# qualityControl1 = np.load(f'../QC/Version_1/{hv}_2018/{hv}_AgloPath_Wei.npy')
# qualityControl2 = np.load(f'../QC/Version_2/{hv}_2018/{hv}_Weight.npy')
# qualityControl3 = np.load(f'../QC/Version_3/{hv}_2018/{hv}_Weight.npy')
# qualityControl4 = np.load(f'../QC/Version_4/{hv}_2018/{hv}_Weight.npy')
i = 33
# Public_Methods.render_Img(ma.array(qualityControl1[i], mask=raw_LAI[i]>70), issave=True, savepath='./Daily_cache/0620/qc1')
# Public_Methods.render_Img(ma.array(qualityControl2[i], mask=raw_LAI[i]>70), issave=True, savepath='./Daily_cache/0620/qc2')
# Public_Methods.render_Img(ma.array(qualityControl3[i], mask=raw_LAI[i]>70), issave=True, savepath='./Daily_cache/0620/qc3')
# Public_Methods.render_Img(ma.array(qualityControl4[i], mask=raw_LAI[i]>70), issave=True, savepath='./Daily_cache/0620/qc4')

# std = 10
# map_std = 0.5 + ((0.15 - 0.5) / (1 - 0)) * (std - 0) 
# print(map_std)

# aa = np.arange(0,6)
# print(aa * 3)
# bb = np.round(5 / aa, 2)
# print(bb)
# cc = np.array(bb)
# print(cc)

# print(np.nan_to_num(cc, posinf=0, neginf=1))