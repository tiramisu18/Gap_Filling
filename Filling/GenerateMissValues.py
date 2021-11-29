# from pyhdf.SD import SD, SDC

# file_name = 'C:\JR_Program\Filling_Missing_Values\MOD15A2H.A2000049.h26v05.006.2015136143539.hdf' 
# file = SD(file_name, SDC.READ) #加载数据

# print(file.info())


import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import copy


def geneMiss(path, data_range):
    # mcd_file_path = 'C:\JR_Program\Filling_Missing_Values\MOD15A2H.A2018361.h27v07.006.2019009094143.hdf' 
    # mcd_file_path = 'C:\JR_Program\MOD15A2H.A2018361.h27v07.006.2019009094143.hdf' 
    file = gdal.Open(path)
    subdatasets = file.GetSubDatasets() #  获取hdf中的子数据集
    # print('Number of subdatasets: {}'.format(len(subdatasets)))
    # print(subdatasets)


 
    # for sd in subdatasets:
    #     print('Name: {0}\nDescription:{1}\n'.format(*sd))

    LAI = gdal.Open(subdatasets[1][0]).ReadAsArray()

    LAI_part = []
    for num1 in range(data_range[0], data_range[1]):
        aa = LAI[num1]
        arr1 = []
        for num2 in range(data_range[0], data_range[1]):
            bb = aa[num2]
            arr1.append(bb)
        LAI_part.append(arr1)
    missing_position1 = np.random.randint(0,100,100)
    missing_position2 = np.random.randint(0,100,100)
    # print(missing_position1, missing_position2)
    LAIPart_backup = copy.deepcopy(LAI_part)
   
    origin_value = []
    for index in range(len(missing_position1)):
        row = LAIPart_backup[missing_position1[index]]
        row[missing_position2[index]] = float('nan')
        origin_value.append(LAI_part[missing_position1[index]][missing_position2[index]])
    
    print(LAIPart_backup)
    # namestr = path.split('\\')[-1].split('.')[1]
    # fig, axs = plt.subplots(1, 2)
    # fig.suptitle(namestr)

    # data = [LAI_part, LAIPart_backup]
    # images = []

    # for i in range(2):
    # # Generate data with a range that varies from one plot to the next.
    #     images.append(axs[i].imshow(data[i], cmap= plt.cm.jet))
    #     axs[i].label_outer()

    # plt.axis('off')
    # # plt.imshow(LAI_part, cmap = 'gray')
    # # plt.imshow(LAIPart_backup, cmap= 'gray') #cmap= plt.cm.jet
    
    # # plt.savefig("./h27v07_test_pic/"+ namestr+".png", dpi=300)
    # plt.show()
    return {'LAI_part': LAI_part, 'LAIPart_backup': LAIPart_backup, 'M_Pos_OriginValue': origin_value, 'missing_pos': [missing_position1, missing_position2]}

# print(origin_value)   
# count = 0
# numlist = []
    # for list in LAI_part: 
    #     for num in list:
    #         if num > 240:
    #             count += 1 
    #             numlist.append(num)
    # if num > 0 and num < 0.228:
    #     count2 = count2 + 1
    # if num > 0.229 and num < 0.445:
    #     count3 = count3 + 1

    # print(count)

# lai_file_path="G:/Postgraduate/LAI_Glass_RTlab/Test_DRT/h20v09.tif"
# mcd_file_path="G:/Postgraduate/LAI_Glass_RTlab/Test_DRT/MCD12Q1.A2018001.h20v09.006.2019199233851.hdf"
# pic_save_path="./"

# for veg_type in range(9):

#     mcd_raster=gdal.Open(mcd_file_path)
#     mcd_sub_dataset=mcd_raster.GetSubDatasets()
#     hdf_band_num=len(mcd_sub_dataset)
#     # for sub_dataset in mcd_sub_dataset:
#     #     print(sub_dataset[1])
#     # print(mcd_sub_dataset[2][1])
#     mcd_sub_type=gdal.Open(mcd_sub_dataset[2][0])
#     mcd_raster_array=mcd_sub_type.ReadAsArray()

#     lai_raster=gdal.Open(lai_file_path)
#     lai_raster_array=lai_raster.ReadAsArray()
#     non_veg_type_lai_array=np.where(mcd_raster_array==veg_type+1,lai_raster_array,np.nan)
#     plt.hist(non_veg_type_lai_array)
#     plt.savefig(pic_save_path+"DRT_"+str(veg_type+1)+".png", dpi=300)
#     plt.clf()
#     plt.cla()