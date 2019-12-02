import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
def default_loader(path):
    img=Image.open(path)
    return img
def default_list_loader(filelist,source):
    imglist=[]
    with open(filelist,'r') as file:
        for line in file.readlines():
            imgpath,id_label=line.strip().rstrip('\n').split(' ')
            imgpath_a,imgpath_b=imgpath.split('/')
            whole_path=os.path.join(source,imgpath_a,imgpath_b)
            imglist.append((whole_path,int(id_label)))
    return imglist
class ImageDataset(data.Dataset):
    def __init__(self,filelist,source,transform1=None,transform2=None,list_loader=default_list_loader,loader=default_loader):
        self.imglist=list_loader(filelist,source)
        self.transform1=transform1
        self.transform2 = transform2
        self.loader=loader
    def __getitem__(self, index):
        imgpath,id_label=self.imglist[index]
        img=self.loader(imgpath)
        a=random.random()
        if a>0.3 :
            img=self.transform1(img)
        else:
            img=self.transform2(img)
        return img,id_label
    def __len__(self):
        return len(self.imglist)

