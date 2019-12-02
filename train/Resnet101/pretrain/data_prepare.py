import os
import shutil
path='/home/guoqiang/reid_competition/train_set/'
txt_path=os.path.join(path,'train_list.txt')
img_path=os.path.join(path,'train_without_augment')
f=open(txt_path)
for line in f.readlines():
    img=line.rstrip().split(' ')[0]
    label=line.rstrip().split(' ')[1]
    pid_dir=os.path.join(img_path,label)
    if not os.path.exists(pid_dir):
        os.makedirs(pid_dir)
    img_name=img[6:]
    img_name_path=os.path.join(img_path,img_name)
    shutil.move(img_name_path,pid_dir)
f.close()