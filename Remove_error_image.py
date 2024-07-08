"""
处理无法打开的图片
"""

import os
from PIL import Image
root_path = r"C:/Users/weiyu/Desktop/photo_org"
root_names = os.listdir(root_path)

for root_name in root_names:
    path = os.path.join(root_path, root_name)
    print("正在删除文件夹：", path)
    names = os.listdir(path)
    names_path = []
    for name in names:
        img = Image.open(os.path.join(path, name))
        name_path = os.path.join(path, name)
        if img == None:
            names_path.append(name_path)
            print("成功保存错误图片路径：{}". format(name))
        else:
            w,h=img.size
            if w<50 or h<50:  # 筛选错误图片
                names_path.append(name_path)
                print("成功保存特小图片路径：{}".format(name))
    print("开始删除需删除的图片")
    for r in names_path:
        os.remove(r)
        print("已删除：", r)
