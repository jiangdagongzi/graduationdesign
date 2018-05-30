from PIL import Image
import os
import os.path

rootdir = r'C:\Users\A\Desktop\1'  # 指明被遍历的文件夹
for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        print('parent is :' + parent)
        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        print('the fulll name of the file is :' + currentPath)

        im = Image.open(currentPath)
        out = im.transpose(Image.FLIP_TOP_BOTTOM)
        newname = r"C:\Users\A\Desktop\1" + '\\' + filename + "(1).jpg"
        out.save(newname)
# im = Image.open(r'C:\Users\Administrator\Desktop\新建文件夹 (2)\1.jpg')
# out = im.transpose(Image.FLIP_LEFT_RIGHT)
# out.save(r'C:\Users\Administrator\Desktop\新建文件夹 (2)\test2.jpg')