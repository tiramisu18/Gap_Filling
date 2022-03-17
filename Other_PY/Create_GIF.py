import imageio
def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=1)#duration间隔
    return

def main():
    print(('begin'))
    image_list = ["/Users/wangjingr/Documents/01-MODIS_Reanalysis/002_Python/Filling/Daily_cache/0316/Simu_Stpe2_LAI/2018_"+str(x)+".png" for x in range(1,47) ]#图片名称列表
    gif_name = '2018_Ori.gif' #输出
    create_gif(image_list, gif_name)



main()