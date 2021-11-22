import imageio
def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=1)#duration间隔
    return

def main():
    print(('begin'))
    image_list = ["/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/Filling/Png_IncludeTitle/h27v07/h27v07_"+str(x)+".png" for x in range(3,45) ]#图片名称列表
    gif_name = 'h27v07_gif.gif' #输出
    create_gif(image_list, gif_name)



main()