from matplotlib import axis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import numpy.ma as ma
from osgeo import gdal
import Public_Methods
import ReadDirFiles

def ReadFile(path):
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()
    return {'LAI': LAI}

# 计算absolute TSS，index范围（1，45）
def cal_TSS(LAIDatas, index, landCover, lcType):
    # LAI_maArray = ma.array(ma.masked_greater(LAIDatas[index, ...], 70), mask=landCover.__ne__(lcType))
    # Public_Methods.render_LAI(LAI_maArray)
    numerators = np.absolute(((LAIDatas[index + 1] - LAIDatas[index - 1]) * index) - (LAIDatas[index] * 2) - ((LAIDatas[index + 1] - LAIDatas[index - 1]) * (index - 1)) + (LAIDatas[index - 1] * 2))
    denominators = np.sqrt(np.square(LAIDatas[index + 1] - LAIDatas[index - 1]) + 2**2)
    absoluteTSS = (numerators / denominators) / 10
    return absoluteTSS

# 计算数据提升之后不同植被类型像元平均年季曲线变化
def landCover_Improved(raw, spatial, temporal, improvedLAI, landCover, lcType): 
    raw_mean = []
    spa_mean= []
    tem_mean = []
    imp_mean = []
    for i in range(0, 46):
        raw_ma = ma.array((ma.masked_greater(raw[i], 70)), mask=(landCover != lcType))
        spa_ma = ma.array((ma.masked_greater(spatial[i], 70)), mask=(landCover != lcType))
        tem_ma = ma.array((ma.masked_greater(temporal[i], 70)), mask=(landCover != lcType))
        imp_ma = ma.array(improvedLAI[i],  mask=landCover != lcType)
        raw_mean.append(ma.mean(raw_ma) / 10)
        spa_mean.append(ma.mean(spa_ma) / 10)
        tem_mean.append(ma.mean(tem_ma) / 10)
        imp_mean.append(ma.mean(imp_ma) / 10)

    Public_Methods.draw_polt_Line(np.arange(0, 361, 8),{
        'title': f'B{lcType}',
        'xlable': 'Day',
        'ylable': 'LAI',
        'line': [raw_mean, spa_mean, tem_mean, imp_mean],
        'le_name': ['Raw','Spatial', 'Temporal', 'Improved'],
        'color': ['gray', '#bfdb39', '#ffe117', '#fd7400'],
        'marker': [',', 'o', '^', '*'],
        'size': {'width': 10, 'height': 6},
        'lineStyle': ['dashed']
        },f'./Daily_cache/0620/lc_part_{lcType}', True, 1)

# 计算单独一期整个tile的LAI差
def diff_LAI(raw, spatial, temporal, i):
    Public_Methods.render_LAI(raw[i], title='LAI', issave=True, savepath='./Daily_cache/0530/RealData/LAI_Raw')
    Public_Methods.render_LAI(spatial[i], title='LAI', issave=True, savepath='./Daily_cache/0530/RealData/LAI_Spatial')
    Public_Methods.render_LAI(temporal[i], title='LAI', issave=True, savepath='./Daily_cache/0530/RealData/LAI_Temporal')

    # 相对标准数据的绝对差
    Spa = (raw[i]-spatial[i]) / 10
    Tem = (raw[i]-temporal[i]) / 10
    # render_Img((StandLAI[i]-InaccurateLAI[i])/10,title='', issave=True, savepath='./Daily_cache/0518/diff_Inaccurate')
    # render_Img((StandLAI[i]-Tem_improvedLAI)/10,title='', issave=True, savepath='./Daily_cache/0518/diff_Tem')
    # render_Img((StandLAI[i]-Spa_improvedLAI)/10,title='', issave=True, savepath='./Daily_cache/0518/diff_Spa')
    data = [Tem, Spa]
    fig, axs = plt.subplots(1, 2)
    # fig.suptitle('Multiple images')
    images = []

    for j in range(2):
            # Generate data with a range that varies from one plot to the next.
            # data = ((1 + i + j) / 10) * np.random.rand(10, 20)
        images.append(axs[j].imshow(data[j], cmap = plt.cm.seismic))
        axs[j].axis('off')
            # axs[i, j].width = 

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = pltcolor.Normalize(vmin=vmin, vmax=vmax,)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs,  fraction=.1)
    plt.savefig('./Daily_cache/0530/RealData/diff_%s'% i, dpi=300)
    plt.show()
    print(np.mean(np.abs(Tem)), np.mean(np.abs(Spa)))
 
def draw_TSS_Image(data, tile, type):
    name = ['Raw', 'Tem', 'Spa']
    for i in range(len(data)):
        plt.imshow(data[i], cmap = plt.cm.rainbow)  # cmap= plt.cm.jet
        plt.title('', family = 'Times New Roman', fontsize = 18)
        colbar = plt.colorbar()
        plt.axis('off')
        plt.savefig('./Daily_cache/0530/TSS/%s_%s_%s' % (type, tile, name[i]), dpi=300)
        plt.show()


lcType = 6
hv = 'h12v04'
# LC
LC_file = gdal.Open(ReadDirFiles.readDir_LC('../LC', hv)[0])
LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()

# Raw LAI
fileLists = ReadDirFiles.readDir(f'../HDF/{hv}')
lai = []
for file in fileLists:
    result = ReadFile(file)
    lai.append(result['LAI'])
raw_LAI = np.array(lai, dtype=float)

qualityControl = np.load(f'../QC/Version_2/{hv}_2018/{hv}_Weight.npy')
# Temporal Spatial LAI
tem = []
spa = []
temWeight = []
spaWeight = []
for i in range(1, 47):
    print(i)
    spa_data = np.load(f'../Improved_RealData/{hv}_2018/Spatial/LAI_{i}.npy')
    tem_data = np.load(f'../Improved_RealData/{hv}_2018/Temporal/LAI_{i}.npy')
    temWei_data = np.load(f'../Improved_RealData/{hv}_2018/Temporal_Weight/LAI_{i}.npy')
    spaWei_data = np.load(f'../Improved_RealData/{hv}_2018/Spatial_Weight/LAI_{i}.npy')
    tem.append(tem_data)
    spa.append(spa_data)
    temWeight.append(temWei_data)
    spaWeight.append(spaWei_data)
tem_LAI = np.array(tem)
spa_LAI = np.array(spa)
tem_Wei = np.array(temWeight)
spa_Wei = np.array(spaWeight)
    
improvedLAI = (ma.masked_greater(tem_LAI, 70) * tem_Wei + ma.masked_greater(spa_LAI, 70) * spa_Wei + ma.masked_greater(raw_LAI, 70) * qualityControl) / (tem_Wei + spa_Wei + qualityControl)
landCover_Improved(raw_LAI, spa_LAI, tem_LAI, improvedLAI, landCover, lcType)
# landCover_Improved(raw_LAI[:,2000:2250, 2000:2250], spa_LAI[:,2000:2250, 2000:2250], tem_LAI[:,2000:2250, 2000:2250], landCover[2000:2250, 2000:2250], lcType)
# Public_Methods.render_LAI(spa_LAI[23], title='', issave=True, savepath='./Daily_cache/0530/RealData/LAI')
# Public_Methods.render_LAI(tem_LAI[23,2000:2250, 2000:2250], title='', issave=True, savepath='./Daily_cache/0530/RealData/Part_LAI')


# 计算绝对TSS
raw_TSS = []
spa_TSS = []
tem_TSS = []
for i in range(1,45):
    print(i)
    raw_one = cal_TSS(raw_LAI, i, landCover, lcType)
    spa_one = cal_TSS(spa_LAI, i, landCover, lcType)
    tem_one = cal_TSS(tem_LAI, i, landCover, lcType)
    # print(one.shape)
    raw_one_ma = ma.array(ma.array(raw_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
    spa_one_ma = ma.array(ma.array(spa_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
    tem_one_ma = ma.array(ma.array(tem_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
    # Public_Methods.render_Img(,issave=True, savepath='./Daily_cache/0530/test%s'% i)
    raw_TSS.append(raw_one_ma)
    spa_TSS.append(spa_one_ma)
    tem_TSS.append(tem_one_ma)

gather_TSS_ab = [ma.array(raw_TSS).sum(axis=0), ma.array(tem_TSS).sum(axis=0), ma.array(spa_TSS).sum(axis=0)]

draw_TSS_Image(gather_TSS_ab, hv, 'absolute')

# 计算相对TSS
gather_TSS_re = [(ma.array(raw_TSS) / (raw_LAI[1:45] / 10)).sum(axis=0), (ma.array(tem_TSS) / (tem_LAI[1:45] / 10)).sum(axis=0), (ma.array(spa_TSS) / (spa_LAI[1:45] / 10)).sum(axis=0)]

draw_TSS_Image(gather_TSS_re, hv, 'relative')

# plt.imshow(raw_TSS[0], cmap = plt.cm.rainbow)  # cmap= plt.cm.jet
# plt.title('', family = 'Times New Roman', fontsize = 18)
# colbar = plt.colorbar()
# # plt.axis('off')
# plt.savefig('./Daily_cache/0530/Test', dpi=300)
# plt.show()

aa = np.arange(12).reshape(2, 3, 2)
print(aa)

print(np.mean(np.mean(aa, axis=1), axis=1))