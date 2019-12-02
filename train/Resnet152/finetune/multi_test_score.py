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
# import pickle
import json
import os.path as osp
from model.model_test_152 import PCB_152
from model.model_test_50 import PCB_50
from model.model_test_101 import PCB_101
from model.resnext50 import PCB_resnext
from utils.re_ranking import re_ranking

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='testing')
parser.add_argument('--test_dir', default='/home/guoqiang/reid_competition/test_set', type=str, help='./test_data')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')

opt = parser.parse_args()

test_dir = opt.test_dir

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,6'

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    # transforms.Resize((432,144),interpolation=3),
    transforms.Resize((384, 128), interpolation=3),
    # transforms.CenterCrop((384,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ############### Five Crop
    # transforms.Lambda(lambda crops: torch.stack(
    #    [transforms.ToTensor()(crop)
    #        for crop in crops]
    #    )),
    # transforms.Lambda(lambda crops: torch.stack(
    #    [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
    #        for crop in crops]
    #   ))
])

data_dir = test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=0) for x in ['gallery', 'query']}

class_names = image_datasets['query'].classes
print(class_names)
use_gpu = torch.cuda.is_available()


# file_name=os.path.basename(__file__)
# file_name=file_name[-5:-3]
# print(file_name)
######################################################################
# Load model
# ---------------------------
def load_network1(network):
    # save_path = os.path.join('./model_gray2',name,'net_%s.pth'%opt.which_epoch)
    save_path = os.path.join('/home/guoqiang/reid_competition/checkpoints/model2/152_81.23.pth')
    network.load_state_dict(torch.load(save_path))
    return network
def load_network2(network):
    # save_path = os.path.join('./model_gray2',name,'net_%s.pth'%opt.which_epoch)
    save_path = os.path.join('/home/guoqiang/reid_competition/checkpoints/resnext.pth')
    network.load_state_dict(torch.load(save_path))
    return network
def load_network3(network):
    # save_path = os.path.join('./model_gray2',name,'net_%s.pth'%opt.which_epoch)
    save_path = os.path.join('/home/guoqiang/reid_competition/checkpoints/model3/101_81.28.pth')
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        # n, ncrops, c, h, w = img.size()
        count += n
        print(count)
        f_part = torch.FloatTensor(n, 3072).zero_()  # the part feature
        f_glo = torch.FloatTensor(n, 2048).zero_()  # the glo feature
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            glo_feature, part_feature = model(input_img)
            # concat each part feature
            feature = torch.FloatTensor()

            for i in range(len(part_feature)):
                temp = part_feature[i]
                # temp = temp.view(n, ncrops, -1).mean(1)
                temp = temp.data.cpu()
                feature = torch.cat((feature, temp), 1)

            # print(f.size())
            f_part = f_part + feature
            f_glo = f_glo + glo_feature.data.cpu()

        f_final = torch.cat((f_part, f_glo), 1)
        # norm feature
        # fnorm = torch.norm(f_final, p=2, dim=1, keepdim=True)    #normalize the feature
        # f_final = f_final.div(fnorm.expand_as(f_final))

        features = torch.cat((features, f_final), 0)  # concat feature according to num
    return features


def cal_cos_dis_2_matrix(a1, a2,a3,b1,b2,b3):

    a=a1
    b=b1
    a_norm = np.sqrt(np.sum(a ** 2, axis=1)).reshape(a.shape[0], 1)
    b_norm = np.sqrt(np.sum(b ** 2, axis=1)).reshape(b.shape[0], 1)
    a_normed=a/a_norm
    b_normed=b/b_norm
    query_feature=a_normed
    gallery_feature=b_normed
    '''
    a_norm = np.sqrt(np.sum(a1 ** 2, axis=1)).reshape(a1.shape[0], 1)
    b_norm = np.sqrt(np.sum(b1 ** 2, axis=1)).reshape(b1.shape[0], 1)
    a1_normed = a1 / a_norm
    b1_normed = b1 / b_norm
    a_norm = np.sqrt(np.sum(a2 ** 2, axis=1)).reshape(a2.shape[0], 1)
    b_norm = np.sqrt(np.sum(b2 ** 2, axis=1)).reshape(b2.shape[0], 1)
    a2_normed = a2 / a_norm
    b2_normed = b2 / b_norm
    query_feature = a1_normed+a2_normed
    gallery_feature = b1_normed+b2_normed
    '''
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
    cos_mat = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    a=a2
    b=b2
    a_norm = np.sqrt(np.sum(a ** 2, axis=1)).reshape(a.shape[0], 1)
    b_norm = np.sqrt(np.sum(b ** 2, axis=1)).reshape(b.shape[0], 1)
    a_normed=a/a_norm
    b_normed=b/b_norm
    query_feature=a_normed
    gallery_feature=b_normed
    '''
    a_norm = np.sqrt(np.sum(a1 ** 2, axis=1)).reshape(a1.shape[0], 1)
    b_norm = np.sqrt(np.sum(b1 ** 2, axis=1)).reshape(b1.shape[0], 1)
    a1_normed = a1 / a_norm
    b1_normed = b1 / b_norm
    a_norm = np.sqrt(np.sum(a2 ** 2, axis=1)).reshape(a2.shape[0], 1)
    b_norm = np.sqrt(np.sum(b2 ** 2, axis=1)).reshape(b2.shape[0], 1)
    a2_normed = a2 / a_norm
    b2_normed = b2 / b_norm
    query_feature = a1_normed+a2_normed
    gallery_feature = b1_normed+b2_normed
    '''
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
    cos_mat2 = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    a=a3
    b=b3
    a_norm = np.sqrt(np.sum(a ** 2, axis=1)).reshape(a.shape[0], 1)
    b_norm = np.sqrt(np.sum(b ** 2, axis=1)).reshape(b.shape[0], 1)
    a_normed=a/a_norm
    b_normed=b/b_norm
    query_feature=a_normed
    gallery_feature=b_normed
    '''
    a_norm = np.sqrt(np.sum(a1 ** 2, axis=1)).reshape(a1.shape[0], 1)
    b_norm = np.sqrt(np.sum(b1 ** 2, axis=1)).reshape(b1.shape[0], 1)
    a1_normed = a1 / a_norm
    b1_normed = b1 / b_norm
    a_norm = np.sqrt(np.sum(a2 ** 2, axis=1)).reshape(a2.shape[0], 1)
    b_norm = np.sqrt(np.sum(b2 ** 2, axis=1)).reshape(b2.shape[0], 1)
    a2_normed = a2 / a_norm
    b2_normed = b2 / b_norm
    query_feature = a1_normed+a2_normed
    gallery_feature = b1_normed+b2_normed
    '''
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
    cos_mat3 = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    cos_mat=(cos_mat+cos_mat2+cos_mat3)/3.0
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


def cal_top_200_similar_ind(a1, a2,a3,b1,b2,b3):
    cos_mat = cal_cos_dis_2_matrix(a1, a2,a3,b1,b2,b3)
    ind_sorted = np.argsort(cos_mat, axis=1)
    sim_ind = ind_sorted[:, 0:200]
    return sim_ind


def load_data(path):
    gal_fea1 = loadmat(path)['gallery_fea1']
    gal_fea2 = loadmat(path)['gallery_fea2']
    gal_fea3 = loadmat(path)['gallery_fea3']
    que_fea1 = loadmat(path)['query_fea1']
    que_fea2 = loadmat(path)['query_fea2']
    que_fea3 = loadmat(path)['query_fea3']

    l_gal = loadmat(path)['gallery_path']
    for index, _ in enumerate(l_gal):
        l_gal[index] = l_gal[index].rstrip()
    l_que = loadmat(path)['query_path']
    for index, _ in enumerate(l_que):
        l_que[index] = l_que[index].rstrip()
    return gal_fea1,gal_fea2, gal_fea3,que_fea1, que_fea2,que_fea3,l_gal, l_que


def get_top_200_similar(sim_ind, l_gal, l_que):
    res_dict = dict()
    for i in range(len(l_que)):
        l = []
        for j in range(sim_ind.shape[1]):
            l.append(l_gal[sim_ind[i, j]])
        res_dict[l_que[i]] = l
    return res_dict


def write_to_json(res_dict, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f)


def compute_result(path, save_path):
    gal_fea1, gal_fea2,gal_fea3,que_fea1,que_fea2, que_fea3,l_gal, l_que = load_data(path)
    sim_ind = cal_top_200_similar_ind(que_fea1, que_fea2,que_fea3,gal_fea1,gal_fea2,gal_fea3)
    res_dict = get_top_200_similar(sim_ind, l_gal, l_que)
    write_to_json(res_dict, save_path)


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
g_path = []
for gp in gallery_path:
    (name, _) = gp
    g_path.append(name.replace(' ', '').split('/')[-1])
q_path = []
for qp in query_path:
    (name, _) = qp
    q_path.append(name.replace(' ', '').split('/')[-1])

######################################################################
# Load Collected data Trained model
print('-------test-----------')
model_structure = PCB_152(class_num=4768)  # for duke
if torch.cuda.device_count() > 1:
    model_structure = nn.DataParallel(model_structure)
model1 = load_network1(model_structure)


model_structure = PCB_resnext(class_num=4768)  # for duke
if torch.cuda.device_count() > 1:
    model_structure = nn.DataParallel(model_structure)
model2 = load_network2(model_structure)

model_structure = PCB_101(class_num=4768)  # for duke

model3 = load_network3(model_structure)
if torch.cuda.device_count() > 1:
    model3 = nn.DataParallel(model3)

# Change to test mode
model1 = model1.eval()
model2 = model2.eval()
model3=model3.eval()

if use_gpu:
    model1 = model1.cuda()
    model2 = model2.cuda()
    model3 = model3.cuda()

# Extract feature
gallery_feature1 = extract_feature(model1, dataloaders['gallery'])
query_feature1 = extract_feature(model1, dataloaders['query'])
gallery_feature2 = extract_feature(model2, dataloaders['gallery'])
query_feature2 = extract_feature(model2, dataloaders['query'])
gallery_feature3 = extract_feature(model3, dataloaders['gallery'])
query_feature3 = extract_feature(model3, dataloaders['query'])

# Save to Matlab for check
result = {'gallery_fea1': gallery_feature1.numpy(), 'gallery_fea2': gallery_feature2.numpy(),'gallery_fea3': gallery_feature3.numpy(),'gallery_path': g_path, 'query_fea1': query_feature1.numpy(),'query_fea2': query_feature2.numpy(),'query_fea3': query_feature3.numpy(),
          'query_path': q_path}
scipy.io.savemat('result.mat', result)

path = 'result.mat'
save_path = 'submission.json'
compute_result(path, save_path)