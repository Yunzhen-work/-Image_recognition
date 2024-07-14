"""
给每一个图片都打上标签
"""

import os
root_path = r"D:/Desktop/photo_org"
save_path = r"D:/Desktop/photo_label"

names = os.listdir(root_path) # 得到image文件夹下的子文件夹的名称
for name in names:
    path = os.path.join(root_path, name)
    img_names = os.listdir(path) # 得到子文件夹下的图片的名称
    for img_name in img_names:
        save_name = img_name.split(".jpg")[0]+'.txt' # 得到相应的lable名称
        txt_path = os.path.join(save_path, name) # 得到lable的子文件夹的路径
        # 结合子文件夹路径和相应文件夹下图片的名称生成相应的子文件夹txt文件
        with open(os.path.join(txt_path, save_name), "w") as f: 
            f.write(name)
            print(f.name)



