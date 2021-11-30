import os

def readDir(dirPath):
    if dirPath[-1] == '/':
        print('path can not end with /')
        return
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        fileList.sort()
        # print(fileList)
        for f in fileList:
            f = dirPath+'/'+f
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles = subFiles + allFiles #合并当前目录与子目录的所有文件路径
            else:
                allFiles.append(f)
        print('allFiles', len(allFiles))
        return allFiles
    else:
        return 'error, not a dir'