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
def landCover_Improved_process(raw, spatial, temporal, improvedLAI, landCover, lcType): 
    raw_mean, spa_mean, tem_mean, imp_mean= [], [], [], []
    for i in range(0, 46):
        raw_ma = ma.array((ma.masked_greater(raw[i], 70)), mask=(landCover != lcType))
        spa_ma = ma.array((ma.masked_greater(spatial[i], 70)), mask=(landCover != lcType))
        tem_ma = ma.array((ma.masked_greater(temporal[i], 70)), mask=(landCover != lcType))
        imp_ma = ma.array((ma.masked_greater(improvedLAI[i], 70)),  mask=landCover != lcType)
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
        'lineStyle': ['solid', 'dashed', 'dashed']
        },f'./Daily_cache/0718/lc_{lcType}_process', True, 1)

def landCover_Improved(raw, improvedLAI, landCover, lcType): 
    raw_mean, imp_mean = [], []
    for i in range(0, 46):
        raw_ma = ma.array((ma.masked_greater(raw[i], 70)), mask=(landCover != lcType))
        imp_ma = ma.array((ma.masked_greater(improvedLAI[i], 70)),  mask=landCover != lcType)
        raw_mean.append(ma.mean(raw_ma) / 10)
        imp_mean.append(ma.mean(imp_ma) / 10)

    Public_Methods.draw_polt_Line(np.arange(0, 361, 8),{
        'title': f'B{lcType}',
        'xlable': 'Day',
        'ylable': 'LAI',
        'line': [raw_mean, imp_mean],
        'le_name': ['Raw', 'Improved'],
        'color': ['#ffe117', '#fd7400'],
        'marker': ['o', '^'],
        'size': {'width': 10, 'height': 6},
        'lineStyle': ['dashed']
        },f'./Daily_cache/0718/Biome_Type_Line/lc_{lcType}_line', True, 1)

def draw_Violinplot(all_data):
    fig, ax = plt.subplots()
  
    vp = ax.violinplot(all_data, [2, 6, 10, 14], widths=2, 
                    showmeans=True, showmedians=True, showextrema=True)
    # styling:
    for body in vp['bodies']:
        body.set_alpha(0.9)
    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #     ylim=(0, 8), yticks=np.arange(1, 8))
    plt.yticks(family='Times New Roman', fontsize=15)
    ax.xaxis.set_visible(False)
    plt.savefig('./Daily_cache/0630/Violinplot', dpi=300)
    plt.show()

# 计算单独一期整个tile的LAI差
# diff_LAI(raw, raw_after, spatial, spatial_n, temporal, temporal_n, average, improved)
def diff_LAI(raw, raw_after, spatial, temporal, improved):
    # Public_Methods.render_LAI(raw, title='LAI', issave=True, savepath='./Daily_cache/0630/LAI_Raw')
    # Public_Methods.render_LAI(raw_after, title='LAI', issave=True, savepath='./Daily_cache/0630/LAI_Raw_after')
    # Public_Methods.render_LAI(spatial, title='LAI', issave=True, savepath='./Daily_cache/0630/LAI_Spatial')
    # Public_Methods.render_LAI(temporal, title='LAI', issave=True, savepath='./Daily_cache/0630/LAI_Temporal')
    # Public_Methods.render_LAI(improved, title='LAI', issave=True, savepath='./Daily_cache/0630/LAI_Improved')

    # 相对标准数据的绝对差
    Self = (raw-raw_after) / 10
    Spa = (raw-spatial) / 10
    # Spa_N = (raw-spatial_n) / 10
    Tem = (raw-temporal) / 10
    # Tem_N = (raw-temporal_n) / 10
    # Ave = (raw-average) / 10
    Imp = (raw-improved) / 10
    # data = [Self, Spa, Spa_N, Tem, Tem_N, Ave, Imp]
    data = [Self, Spa, Tem, Imp]
    count = 4
    fig, axs = plt.subplots(1, count)
    # fig.suptitle()
    images = []
    for j in range(count):
            # Generate data with a range that varies from one plot to the next.
            # data = ((1 + i + j) / 10) * np.random.rand(10, 20)
        images.append(axs[j].imshow(data[j], cmap = plt.cm.seismic))
        axs[j].axis('off')
        # axs[j].width = 100

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = pltcolor.Normalize(vmin=vmin, vmax=vmax,)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs,  fraction=.1)
    plt.savefig('./Daily_cache/0630/diff', dpi=300)
    plt.show()
    # print(np.mean(np.abs(Self)), np.mean(np.abs(Spa)),np.mean(np.abs(Spa_N)),np.mean(np.abs(Tem)),np.mean(np.abs(Tem_N)),  np.mean(np.abs(Ave)), np.mean(np.abs(Imp)))
    print(np.mean(np.abs(Self)), np.mean(np.abs(Spa)),np.mean(np.abs(Tem)), np.mean(np.abs(Imp)))
    draw_Violinplot([Self.flatten(), Spa.flatten(), Tem.flatten(), Imp.flatten()])

def draw_TSS_Image(data, tile, type):
    name = ['Raw', 'Temporal', 'Spatial', 'Improved']
    # # data = [Self, Spa, Tem, Imp]
    # count = 4
    # fig, axs = plt.subplots(1, count, figsize=(15,8))
    # # fig.suptitle()
    # images = []
    # for j in range(count):
    #         # Generate data with a range that varies from one plot to the next.
    #         # data = ((1 + i + j) / 10) * np.random.rand(10, 20)
    #     images.append(axs[j].imshow(data[j], cmap = plt.cm.seismic))
    #     axs[j].axis('off')
    #     # axs[j].width = 100

    # # Find the min and max of all colors for use in setting the color scale.
    # vmin = min(image.get_array().min() for image in images)
    # vmax = max(image.get_array().max() for image in images)
    # norm = pltcolor.Normalize(vmin=vmin, vmax=vmax,)
    # for im in images:
    #     im.set_norm(norm)

    # fig.colorbar(images[0], ax=axs,  fraction=.1)
    # plt.savefig('./Daily_cache/0630/TSS/%s_%s_all' % (type, tile), dpi=300)
    # plt.show()

    for i in range(len(data)):
        plt.imshow(data[i], cmap = plt.cm.rainbow)  # cmap= plt.cm.jet
        plt.title('', family = 'Times New Roman', fontsize = 18)
        colbar = plt.colorbar()
        plt.axis('off')
        plt.savefig('./Daily_cache/0630/TSS/%s_%s_%s' % (type, tile, name[i]), dpi=300)
        plt.show()


lcType = 4
hv = 'h12v03'
url = f'../Improved/Improved_RealData/{hv}_2018'
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
# raw_LAI = np.array(lai, dtype=float)[:, 1450:1650, 1300:1500]

# index = 6
# diff_LAI(raw_LAI[index], 
#     raw_LAI[index+1], 
#     np.load(f'{url}/Spatial/LAI_{index+2}.npy')[50:250, 50:250], 
#     # np.load(f'{url}/Spatial_N/LAI_{index+2}.npy')[50:250, 50:250], 
#     np.load(f'{url}/Temporal/LAI_{index+2}.npy')[50:250, 50:250], 
#     # np.load(f'{url}/Temporal_N/LAI_{index+2}.npy')[50:250, 50:250], 
#     # (np.load(f'{url}/Temporal/LAI_{index+2}.npy')[50:250, 50:250] + np.load(f'{url}/Spatial/LAI_{index+2}.npy')[50:250, 50:250]) / 2, 
#     np.load(f'{url}/Improved/LAI_{index+2}.npy')[50:250, 50:250])


# Temporal Spatial Improved LAI
# spa_LAI, tem_LAI, imp_LAI = [], [], []
# for i in range(1, 47):
#     print(i)
#     spa_LAI.append(np.load(f'{url}/Spatial/LAI_{i}.npy'))
#     tem_LAI.append(np.load(f'{url}/Temporal/LAI_{i}.npy'))
#     imp_LAI.append(np.load(f'{url}/Improved/LAI_{i}.npy'))
# spa_LAI = np.array(spa_LAI)
# tem_LAI = np.array(tem_LAI)
# imp_LAI = np.array(imp_LAI)
# landCover_Improved_process(raw_LAI, spa_LAI, tem_LAI, imp_LAI, landCover, lcType)

# Improved LAI
imp_LAI = []
for i in range(1, 47):
    imp_LAI.append(np.load(f'{url}/Improved/LAI_{i}.npy'))
imp_LAI = np.array(imp_LAI)
landCover_Improved(raw_LAI, imp_LAI, landCover, lcType)

# 计算绝对TSS
raw_TSS = []
spa_TSS = []
tem_TSS = []
imp_TSS = []
for i in range(1,45):
    print(i)
    raw_one = cal_TSS(raw_LAI, i, landCover, lcType)
    spa_one = cal_TSS(spa_LAI, i, landCover, lcType)
    tem_one = cal_TSS(tem_LAI, i, landCover, lcType)
    imp_one = cal_TSS(imp_LAI, i, landCover, lcType)
    # print(one.shape)
    raw_one_ma = ma.array(ma.array(raw_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
    spa_one_ma = ma.array(ma.array(spa_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
    tem_one_ma = ma.array(ma.array(tem_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
    imp_one_ma = ma.array(ma.array(imp_one, dtype='float', mask=raw_LAI[i].__gt__(70)), mask=landCover.__ne__(lcType))
    # Public_Methods.render_Img(,issave=True, savepath='./Daily_cache/0530/test%s'% i)
    raw_TSS.append(raw_one_ma)
    spa_TSS.append(spa_one_ma)
    tem_TSS.append(tem_one_ma)
    imp_TSS.append(imp_one_ma)

raw_Ab_Gather = ma.array(raw_TSS).sum(axis=0)
tem_Ab_Gather = ma.array(tem_TSS).sum(axis=0)
spa_Ab_Gather = ma.array(spa_TSS).sum(axis=0)
imp_Ab_Gather = ma.array(imp_TSS).sum(axis=0)

draw_TSS_Image([raw_Ab_Gather, tem_Ab_Gather, spa_Ab_Gather, imp_Ab_Gather], hv, 'absolute')

# 计算相对TSS
raw_Re_Gather = (ma.array(raw_TSS) / (raw_LAI[1:45] / 10)).sum(axis=0)
tem_Re_Gather = (ma.array(tem_TSS) / (tem_LAI[1:45] / 10)).sum(axis=0)
spa_Re_Gather = (ma.array(spa_TSS) / (spa_LAI[1:45] / 10)).sum(axis=0)
imp_Re_Gather = (ma.array(imp_TSS) / (imp_LAI[1:45] / 10)).sum(axis=0)


# 绘制误差的分布密度直方图
# h1 = (tem_LAI - raw_LAI).flatten()
# h2 = (spa_LAI - raw_LAI).flatten()
# h3 = (imp_LAI - raw_LAI).flatten()
# fig, ax = plt.subplots(figsize=(10,5))
# # ax.hist(raw_Re_Gather.flatten(), density=True, histtype="stepfilled", bins=50, alpha=0.8, label='Raw')
# ax.hist(h1, density=True, histtype="stepfilled", bins=50, alpha=0.6, label='Temporal')
# ax.hist(h2, density=True, histtype="stepfilled", bins=50, alpha=0.6, label='Spatial')
# ax.hist(h3, density=True, histtype="stepfilled", bins=50, alpha=0.6, label='Improved')

    
# ax.set_xlabel('Relative TSS', fontsize=15, family='Times New Roman')
# ax.set_ylabel('Density', fontsize=15, family='Times New Roman')
# ax.legend(prop={'size':15, 'family':'Times New Roman'})
# fig.tight_layout()
# # ax.set(xlim=(0, 500))
#         # ylim=(0, 8), yticks=np.arange(1, 8))
# plt.xticks( family='Times New Roman', fontsize=15)
# plt.yticks( family='Times New Roman', fontsize=15)
# plt.savefig('./Daily_cache/0630/histogram', dpi=300)
# plt.show()

