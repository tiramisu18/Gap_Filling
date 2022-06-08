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

aa = readTif('../GBOV/GBOV_LP03_L08_BART_20180516_20180516_001_UOS_V3.0_300M.TIF').ReadAsArray()

# print(ma.mean(ma.masked_equal(aa[0], 0)))
print(np.mean(aa[0]))
print(aa[0])

render_Img(aa[0])

render_Img(aa[1])
render_Img(aa[2])

render_Img(aa[3])
render_Img(aa[4])