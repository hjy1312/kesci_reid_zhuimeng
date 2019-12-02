import glob
import shutil
import os
import datetime
time1=datetime.datetime.now()
time1_str=datetime.datetime.strftime(time1,'%Y-%m-%d_%H:%M:%S')
script_set1=glob.glob('./*.py')
script_set2=glob.glob('./utils/*.py')+glob.glob('./model/*.py')
saved_folder=os.path.join('./saved_folder',time1_str)
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)
for path in script_set1:
    save_path=os.path.join(saved_folder,os.path.split(path)[-1])
    shutil.copyfile(path,save_path)
for path in script_set2:
    folder_path=os.path.join(saved_folder,path.split('/')[-2])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    save_path=os.path.join(folder_path,path.split('/')[-1])
    shutil.copyfile(path, save_path)