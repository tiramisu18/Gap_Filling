import numpy as np
import numpy.ma as ma
from osgeo import gdal
import matplotlib.pyplot as plt

def render_Img (data, title='', issave=False, savepath='', color=plt.cm.jet):
    plt.imshow(data, cmap = color)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    # plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()


def readFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    # QC = gdal.Open(subdatasets[2][0]).ReadAsArray()
    return LAI


def calculate_RawMean(fileLists, index, line, samp):
    data = readFile(fileLists[index])
    step1 = data[line-2:line+4, samp-2:samp+4]
    step2 = ma.masked_greater(step1, 70)
    return ma.mean(step2)

# 计算整个Tile内的6*6
def calculate_TemSpaMean(url, line, samp):
    data = np.load(url)
    step1 = data[line-2:line+4, samp-2:samp+4]
    step2 = ma.masked_greater(step1, 70)
    return ma.mean(step2)

# 计算站点范围21*21内的6*6
def calculate_part(url):
    data = np.load(url)
    step1 = data[8:14, 8:14]
    step2 = ma.masked_greater(step1, 70)
    return ma.mean(step2)

