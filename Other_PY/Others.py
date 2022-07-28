# import matplotlib.pyplot as plt
# import numpy as np

# # plt.style.use('_mpl-gallery')

# # make data
# np.random.seed(1)
# x = np.linspace(0, 8, 16)
# y1 = 3 + 4*x/8 + np.random.uniform(0.0, 0.5, len(x))
# y2 = 1 + 2*x/8 + np.random.uniform(0.0, 0.5, len(x))
# print(x)
# # plot
# fig, ax = plt.subplots()

# ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
# # ax.plot(x, (y1 + y2)/2, linewidth=2)

# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

# plt.show()


# from matplotlib import colors
# import matplotlib.pyplot as plt
# import numpy as np

# np.random.seed(19680801) #使每次运行结果一致
# Nr = 3
# Nc = 2

# fig, axs = plt.subplots(Nr, Nc)
# # fig.suptitle('Multiple images')

# images = []
# for i in range(Nr):
#     for j in range(Nc):
#         # Generate data with a range that varies from one plot to the next.
#         data = ((1 + i + j) / 10) * np.random.rand(10, 20)
#         images.append(axs[i, j].imshow(data))
#         axs[i, j].label_outer()

# # Find the min and max of all colors for use in setting the color scale.
# vmin = min(image.get_array().min() for image in images)
# vmax = max(image.get_array().max() for image in images)
# norm = colors.Normalize(vmin=vmin, vmax=vmax)
# for im in images:
#     im.set_norm(norm)

# fig.colorbar(images[0], ax=axs,  fraction=.1)

# plt.show()


# 生成散点图的图例
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('_mpl-gallery')

# make the data
np.random.seed(3)
x = 1 + np.random.normal(0, 0, 10)
y = 1 + np.random.normal(0, 0, len(x))
# size and color:
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots(figsize=(7,4))
ax.scatter(x, y, c='#53aead', s=50, marker='^', label='Raw')
ax.scatter(x, y, c='#53aead', s=100, marker='*', label='Improved')
colorsPara = {'Forest':'#e85c46', 'Crops':'#53aead', 'Crops & Savannas':'#bee5a0', 'Forest & Savannas': '#fdbe6f'}

# for key, color in colorsPara.items():
#     ax.scatter(x, y, s=30, marker='^', c=color, vmin=0, vmax=100, label=key)   

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))
plt.legend(loc = 2, framealpha=0,  bbox_to_anchor=(0.2, 0.75), markerscale=2, labelspacing=0.7, handletextpad=0.4
        ,prop={'size':20, 'family':'Times New Roman'})
plt.savefig('./PNG/Scatter_legend_2', dpi=300)
plt.show()