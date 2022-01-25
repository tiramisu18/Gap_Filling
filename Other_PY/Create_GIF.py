import imageio
def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=1)#duration间隔
    return

def main():
    print(('begin'))
    image_list = ["/Users/wangjingr/Documents/01-MODIS_Reanalysis/002_Python/QC/Img/h11v04_2018_CloudState/h11v04_2018_"+str(x)+".png" for x in range(1,47) ]#图片名称列表
    gif_name = 'h11v04_2018_CloudState.gif' #输出
    create_gif(image_list, gif_name)



main()