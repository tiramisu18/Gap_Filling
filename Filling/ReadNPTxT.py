import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor


colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7', '#d7efb3', '#a57d78', '#8e8681']
bounds = [0,10,20,30,40,50,60,70,250]
cmap = pltcolor.ListedColormap(colors)
norm = pltcolor.BoundaryNorm(bounds, cmap.N)
file_hv = 'h27v07'


# #render PNG
for index in range(3, 45):
  data_array = np.loadtxt('/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/Original_Data/' + file_hv + '/' + file_hv + '_' + str(index))
  plt.title('2018 ' + str(1 + 8 * index) , family='Times New Roman', fontsize=18)
  plt.imshow(data_array, cmap=cmap, norm=norm)
  cbar = plt.colorbar()
  plt.axis('off')
  cbar.set_ticklabels(['0','1','2','3','4','5','6','7'])
  plt.savefig('/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/Filling/Png_IncludeTitle/' + file_hv + '/' + file_hv + '_' + str(index) + '.png', dpi=300)
# 去除图片空白区域
#   fig = plt.gcf()
#   fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
#   plt.gca().xaxis.set_major_locator(plt.NullLocator())
#   plt.gca().yaxis.set_major_locator(plt.NullLocator())
#   plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#   plt.margins(0,0)
#   plt.savefig("/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/Original_Data/PNG/h27v07/h27v07_" + str(index) + ".png", dpi=300)
  plt.show()

# F_all_mean = []
# O_all_mean = []
# for index in range(3,45):
#   print(index)
#   data_array_F = np.loadtxt('/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/Results/Data/' + file_hv + '/' + file_hv + '_' + str(index))
#   data_one_dimensional_F = data_array_F.flatten()
#   data_array_O = np.loadtxt('/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/Original_Data/' + file_hv + '/' + file_hv + '_' + str(index))
#   data_one_dimensional_O = data_array_O.flatten()
#   data_valid_F = []
#   data_valid_O = []
#   for ele in range(len(data_one_dimensional_F)):
#     if data_one_dimensional_F[ele] <= 70: data_valid_F.append(data_one_dimensional_F[ele])
#     if data_one_dimensional_O[ele] <= 70: data_valid_O.append(data_one_dimensional_O[ele])

#   # print(data_valid)
#   data_mean_F = np.mean(data_valid_F)
#   data_mean_O = np.mean(data_valid_O)
#   # print(len(data_valid), data_mean)
#   F_all_mean.append(data_mean_F)
#   O_all_mean.append(data_mean_O)
# # line chart
# aa = np.arange(25, 361, 8)
# # plt.xticks(aa)
# plt.figure(figsize=(15, 6)) #宽，高
# plt.title(file_hv, family='Times New Roman', fontsize=18)
# plt.xlabel('Day', fontsize=15, family='Times New Roman') 
# plt.ylabel('Value', fontsize=15, family='Times New Roman')

# line1=plt.plot(aa,O_all_mean, label='count', color='#fd7400',  marker='o', markersize=5)
# line4=plt.plot(aa,F_all_mean, label='count',color='#bfdb39',  marker='o', markersize=5)
# plt.legend(
#   (line1[0],  line4[0]), 
#   ('Original', "Filling"),
#   loc = 1, prop={'size':15, 'family':'Times New Roman'},
#   )
# plt.savefig(file_hv + '_line_1027.png', dpi=300)
# plt.show()

