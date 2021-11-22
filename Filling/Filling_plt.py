import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor

image_data = [[],[]]
colors = ['#016382', '#1f8a6f', '#bfdb39', '#ffe117', '#fd7400', '#e1dcd7','#d7efb3', '#a57d78', '#8e8681']
bounds = [0,10,20,30,40,50,60,70,250]
cmap = pltcolor.ListedColormap(colors)
norm = pltcolor.BoundaryNorm(bounds, cmap.N)
plt.title('title', family='Times New Roman', fontsize=18)
plt.imshow(image_data, cmap=cmap, norm=norm)
cbar = plt.colorbar()
cbar.set_ticklabels(['0','10','20','30','40','50','60','70','250'])
# plt.savefig("./h27v07_test_pic/Origin_part.png", dpi=300)
plt.show()