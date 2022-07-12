from osgeo import gdal
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from Filling import Improved_Pixel, ReadDirFiles, Public_Methods

sitesinfo = {
    'BART': {'h': 12, 'v': 4, 'line': 1424.16, 'samp': 2105.61},
    'BLAN': {'h': 11, 'v': 5, 'line': 225.04, 'samp': 2250.38},
    'CPER': {'h': 10, 'v': 4, 'line': 2203.77, 'samp': 173.89},
    'DSNY': {'h': 10, 'v': 6, 'line': 449.49, 'samp': 1962.62},
    'DELA': {'h': 10, 'v': 5, 'line': 1789.49, 'samp': 1435.02},
    'GUAN': {'h': 11, 'v': 7, 'line': 486.81, 'samp': 1533.85},
    'HARV': {'h': 12, 'v': 4, 'line': 1790.43, 'samp': 1636.72},
    'JERC': {'h': 10, 'v': 5, 'line': 2112.74, 'samp': 1858.18},
    'JORN': {'h': 8, 'v': 5, 'line': 1777.73, 'samp': 2394.90},
    'KONA': {'h': 10, 'v': 5, 'line': 212.99, 'samp': 1207.90},
    'LAJA': {'h': 11, 'v': 7, 'line': 474.40, 'samp': 1490.80},
    'MOAB': {'h': 9, 'v':5, 'line': 419.89, 'samp': 981.96},
    'NIWO': {'h': 9, 'v':4, 'line': 2386.47, 'samp': 2203.54},
    'ONAQ': {'h': 9, 'v':4, 'line': 2356.88, 'samp': 978.91},
    'ORNL': {'h': 11, 'v':5, 'line': 968.11, 'samp': 427.40},
    'OSBS': {'h': 10, 'v':6, 'line': 77.22, 'samp': 2099.01},
    'SCBI': {'h': 11, 'v':5, 'line': 265.20, 'samp': 2203.28},
    'SERC': {'h': 12, 'v':5, 'line': 265.86, 'samp': 97.75},
    'STEI': {'h': 11, 'v':4, 'line': 1077.35, 'samp': 1731.83},
    'STER': {'h': 10, 'v':4, 'line': 2288.63, 'samp': 386.25},
    'SRER': {'h': 8, 'v':5, 'line': 1940.94, 'samp': 1419.03},
    'TALL': {'h': 10, 'v':5, 'line': 1691.39, 'samp': 1599.03},
    'UNDE': {'h': 11, 'v':4, 'line': 903.35, 'samp': 1935.23},
    'WOOD': {'h': 11, 'v':4, 'line': 688.72, 'samp': 594.74},
} # 24个

sites, h, v, lines, samps, landcovers = [], [], [], [], [], []

for key, ele in sitesinfo.items():
    hv = 'h%02dv%02d' % (ele['h'], ele['v'])
    line = int(ele['line'])
    samp = int(ele['samp'])
    sites.append(key)
    h.append(ele['h'])
    v.append(ele['v'])
    lines.append(ele['line'])
    samps.append(ele['samp'])

    LC_file = gdal.Open(ReadDirFiles.readDir_LC('../LC', hv)[0])
    LC_subdatasets = LC_file.GetSubDatasets()  # 获取hdf中的子数据集
    landCover = gdal.Open(LC_subdatasets[2][0]).ReadAsArray()[line-2:line+4, samp-2:samp+4]
    landcovers.append(landCover)

info = pd.DataFrame({'Site': sites, 'h': h, 'v': v, 'line': lines, 'samp': samps, 'landcover': landcovers })
    # print(specific)
info.to_csv(f'./Site_Info.csv')