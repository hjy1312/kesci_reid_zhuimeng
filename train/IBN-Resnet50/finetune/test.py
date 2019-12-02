# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import glob
import numpy as np
from scipy.io import loadmat
#import pickle
import json
import os.path as osp
import sys
sys.path.append('./model')
from model import PCB
#from model.model_test_152 import PCB_152
from utils.re_ranking import re_ranking

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='testing')
parser.add_argument('--test_dir',default='./test_set',type=str, help='./test_data')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')

opt = parser.parse_args()



test_dir = opt.test_dir

os.environ["CUDA_VISIBLE_DEVICES"] ='2'

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        #transforms.Resize((432,144),interpolation=3),
        transforms.Resize((384,128), interpolation=3),
        #transforms.CenterCrop((384,128)),        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Five Crop        
        #transforms.Lambda(lambda crops: torch.stack(
        #    [transforms.ToTensor()(crop) 
        #        for crop in crops]
        #    )),
        #transforms.Lambda(lambda crops: torch.stack(
        #    [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
        #        for crop in crops]
        #   ))
])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=0) for x in ['gallery','query']}

class_names = image_datasets['query'].classes
print(class_names)
use_gpu = torch.cuda.is_available()
#file_name=os.path.basename(__file__)
#file_name=file_name[-5:-3]
#print(file_name)
######################################################################
# Load model
#---------------------------
def load_network(network):
    #save_path = os.path.join('./model_gray2',name,'net_%s.pth'%opt.which_epoch)
    save_path = os.path.join('/home/junyang/experiment/re-id-qiang/checkpoints/model_2019-11-25 09:40:19/net_59.pth')
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Extract feature
# ----------------------

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        #n, ncrops, c, h, w = img.size()
        count += n
        print(count)
        f_part = torch.FloatTensor(n, 3072).zero_() # the part feature
        f_glo = torch.FloatTensor(n, 2048).zero_ () # the glo feature
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            glo_feature,part_feature = model(input_img)
            # concat each part feature 
            feature = torch.FloatTensor()

            for i in range(len(part_feature)):
                temp = part_feature[i]
                # temp = temp.view(n, ncrops, -1).mean(1)
                temp = temp.data.cpu()
                feature = torch.cat((feature, temp), 1)
                
            #print(f.size())
            f_part = f_part + feature
            f_glo = f_glo + glo_feature.data.cpu()

        f_final = torch.cat((f_part, f_glo), 1)
        # norm feature
        #fnorm = torch.norm(f_final, p=2, dim=1, keepdim=True)    #normalize the feature
        #f_final = f_final.div(fnorm.expand_as(f_final))
        
        features = torch.cat((features,f_final), 0)  # concat feature according to num
    return features

def cal_cos_dis_2_matrix(a,b):
    a_norm = np.sqrt(np.sum(a**2,axis=1)).reshape(a.shape[0],1)
    b_norm = np.sqrt(np.sum(b**2,axis=1)).reshape(b.shape[0],1)
    a_normed = a / a_norm
    b_normed = b / b_norm
    query_feature=a_normed
    gallery_feature=b_normed
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
    cos_mat = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    return cos_mat
'''
def cal_cos_dis_2_matrix(qf,gf):
    qf=torch.FloatTensor(qf)
    gf=torch.FloatTensor(gf)
    m, n = qf.size(0), gf.size(0)
    qf_norm=torch.sqrt(torch.sum(torch.pow(qf,2),dim=1)).unsqueeze(1).expand_as(qf)
    gf_norm = torch.sqrt(torch.sum(torch.pow(gf, 2), dim=1)).unsqueeze(1).expand_as(gf)
    qf=torch.div(qf,qf_norm)
    gf=torch.div(gf,gf_norm)

    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    return distmat
'''
def cal_top_200_similar_ind(a,b):
    cos_mat = cal_cos_dis_2_matrix(a,b)
    ind_sorted = np.argsort(cos_mat,axis=1)
    sim_ind = ind_sorted[:,0:200]
    return sim_ind

def load_data(path):
    gal_fea = loadmat(path)['gallery_fea']
    que_fea = loadmat(path)['query_fea']
    l_gal = loadmat(path)['gallery_path']
    for index,_ in enumerate(l_gal):
        l_gal[index]=l_gal[index].rstrip()
    l_que = loadmat(path)['query_path']
    for index,_ in enumerate(l_que):
        l_que[index]=l_que[index].rstrip()
    return gal_fea,que_fea,l_gal,l_que

def get_top_200_similar(sim_ind,l_gal,l_que):
    res_dict = dict()
    for i in range(len(l_que)):
        l = []
        for j in range(sim_ind.shape[1]):
            l.append(l_gal[sim_ind[i,j]])
        res_dict[l_que[i]] = l
    return res_dict

def write_to_json(res_dict,save_path):
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(res_dict,f)

def compute_result(path,save_path):
    gal_fea,que_fea,l_gal,l_que = load_data(path)
    sim_ind = cal_top_200_similar_ind(que_fea,gal_fea)
    res_dict = get_top_200_similar(sim_ind,l_gal,l_que)
    write_to_json(res_dict,save_path)

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
g_path=[]
for gp in gallery_path:
    (name,_)=gp
    g_path.append(name.replace(' ','').split('/')[-1])
q_path=[]
for qp in query_path:
    (name,_)=qp
    q_path.append(name.replace(' ','').split('/')[-1])




######################################################################
# Load Collected data Trained model
print('-------test-----------')
model_structure = PCB(class_num=4768) #for duke
if torch.cuda.device_count() > 1:
    model_structure=nn.DataParallel(model_structure)
model = load_network(model_structure)


# Change to test mode
model = model.eval()

if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
query_feature = extract_feature(model,dataloaders['query'])

# Save to Matlab for check
result = {'gallery_fea':gallery_feature.numpy(),'gallery_path':g_path,'query_fea':query_feature.numpy(),'query_path':q_path}
scipy.io.savemat('result.mat',result)

path = 'result.mat'
save_path = 'submission.json'
compute_result(path,save_path)
