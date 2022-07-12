import numpy as np
import numpy.ma as ma
from osgeo import gdal
import sys
sys.path.append("..")
from Filling import Improved_Pixel, Public_Methods, ReadDirFiles

# 读取HDF文件数据集
def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    StdLAI = gdal.Open(subdatasets[5][0]).ReadAsArray()
    return {'LAI': LAI, 'QC': QC, 'StdLAI': StdLAI}

# 权重加上StdLAI和相对TSS
def addStdLAITSS(StdLAIDatas, TSSValues, hv, ):
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
    return final_qualityControl


def cal_TSS(LAIDatas, index):
    numerators = np.absolute(((LAIDatas[index + 1] - LAIDatas[index - 1]) * index) - (LAIDatas[index] * 2) - ((LAIDatas[index + 1] - LAIDatas[index - 1]) * (index - 1)) + (LAIDatas[index - 1] * 2))
    denominators = np.sqrt(np.square(LAIDatas[index + 1] - LAIDatas[index - 1]) + 2**2)
    # absoluteTSS = (numerators / denominators) / 10
    absoluteTSS = numerators / denominators
    relativeTSS = np.round(absoluteTSS / LAIDatas[index], 2)
    return np.nan_to_num(relativeTSS, posinf=10, neginf=10)
  
hv = 'h12v04'
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


addStdLAITSS(StdLAIDatas, TSSArray, hv, f'../QC/Version_4/{hv}_2018')

