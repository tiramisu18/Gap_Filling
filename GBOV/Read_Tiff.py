from osgeo import gdal
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
#读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
    return dataset

def render_Img (data, title='', issave=False, savepath='', color=plt.cm.jet):
    plt.imshow(data, cmap = color)  # cmap= plt.cm.jet
    plt.title(title, family='Times New Roman', fontsize=18)
    colbar = plt.colorbar()
    # plt.axis('off')
    if issave :plt.savefig(savepath, dpi=300)
    plt.show()

def calculate_Mean(url):
    aa = readTif(url).ReadAsArray()

    # print(ma.mean(ma.masked_equal(aa[0], 0)))
    step1 = ma.fix_invalid(aa[0])
    step2 = ma.array(step1, mask=aa[2].__eq__(1))
    step3 = ma.array(step2, mask=aa[3].__eq__(1))
    return ma.mean(step3)
    # print(ma.mean(step1))
    # render_Img(aa[0])
    # # print(step1)
    # step2 = ma.array(step1, mask=aa[3].__eq__(1))
    # print(ma.mean(step2))
    # render_Img(step2)

    # step3 = ma.array(step2, mask=aa[2].__eq__(1))
    # print(ma.mean(step3))
    # render_Img(step3)

    # render_Img(aa[1])
    # render_Img(aa[2])
    # render_Img(aa[3])

    # render_Img(aa[4])

calculate_Mean('./Raw_Data/Test/GBOV_LP03_L08_BART_20170326_20170326_001_UOS_V3.0_300M.TIF')