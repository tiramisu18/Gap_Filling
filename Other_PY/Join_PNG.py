from PIL import Image


def join(png1, png2, spacing, save_position, flag='horizontal'):
    """
    :param png1: png1 path 
    :param png2: png2 path
    :param spacing: spacing width, default 0
    :param save_position: save image
    :param flag: horizontal or vertical
    :return:
    """
    img1, img2 = Image.open(png1), Image.open(png2)
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        joint = Image.new('RGB', (size1[0]+size2[0] + spacing, size1[1]))
        loc1, loc2 = (0, 0), (size1[0] + spacing, 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_position)
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1]+size2[1] + spacing))
        loc1, loc2 = (0, 0), (0, size1[1] + spacing)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_position)

# img1 = './h27v07_png/h27v07_3.png'
# img2 = './h28v07_png/h28v07_3.png'
# join(img1, img2, 0, './merge_3_1.png')
img1 = '/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/h27v07_43.png'
img2 = '/Users/wangjingr/Documents/01-MODIS Reanalysis/002_Python/h27v07_44.png'
join(img1, img2, 0, './h27v07_43_merge.png', 'horizontal')


# png1和png2分别是两张图片的位置
# spacing控制着两张图片拼接的距离，即通过控制这个让两张图像不至于靠的太近。
# save_position 文件保存位置
# flag控制水平拼接还是垂直拼接，默认是水平拼接。