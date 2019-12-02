# -*- coding: utf-8 -*-

from __future__ import print_function, division
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import os
import os.path as osp
from torch.nn import init
import sys
sys.path.append('./model')
from model import PCB
from utils.random_erasing import RandomErasing
from utils.batch_sample import BalancedBatchSampler
from utils.triplet_sampling import HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector
from utils.losses import OnlineTripletLoss
from ImageDataset import ImageDataset
from utils.labelsmoothing import CrossEntropyLabelSmooth
from utils.centerloss import CenterLoss
from utils.focalloss import  FocalLoss
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir', default='./train_set/train', type=str,
                    help='training dir path')
parser.add_argument('--color_jitter', default=False, help='use color jitter in training')
parser.add_argument('--erasing_p', default=0.0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--outf', default='.', help='folder to output model checkpoints')
parser.add_argument('--n_classes', default=32, type=int, help='how many person in a batch')
parser.add_argument('--n_images', default=2, type=int, help='how many image for each person in a batch')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--margin', default=0.3, type=float, help='margin of triplet')

opt = parser.parse_args()
data_dir = opt.data_dir
time1 = datetime.datetime.now()
time1_str = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
time1_str.replace(' ','_')
opt.outf = './checkpoints/model_'+time1_str
try:
    os.makedirs(opt.outf)
except OSError:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

######################################################################
# Load Data
# ---------

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((384, 128), interpolation=3),  # resize
    transforms.RandomApply([transforms.Pad(10),\
                            transforms.RandomCrop((384, 128))], p=0.5),
    # transforms.RandomGrayscale(p=0.2),
    # transforms.RandomCrop((256,128)),
    transforms.RandomHorizontalFlip(),  # randomly horizon flip image
    transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # [m1,m2...mn][s1,s2...sn] for n channels
]
transform_train_list2 = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((384, 128), interpolation=3),  # resize
    # transforms.RandomGrayscale(p=0.2),
    # transforms.RandomCrop((256,128)),
    # transforms.RandomHorizontalFlip(),#randomly horizon flip image
    transforms.ToTensor(),  # convert PIL image or numpy.ndarray to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # [m1,m2...mn][s1,s2...sn] for n channels
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(
        opt.erasing_p)]  # randomly select a rectangle region in a image and erase its pixels with random values
if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list
# randomly change the brightness,contrast,saturation
print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),  # compose several transforms together
    'none': transforms.Compose(transform_train_list2)
}

image_datasets = datasets.ImageFolder(os.path.join(data_dir), data_transforms[
    'train'])  # return several image folders indicating class including images
# image_datasets=ImageDataset(filelist='./train_set/train_list.txt',source='./train_set',transform1=data_transforms['train'],transform2=data_transforms['none'])
print(len(image_datasets))
# batch sampling for dataset
Sampler = BalancedBatchSampler(image_datasets, n_classes=opt.n_classes, n_samples=opt.n_images)

dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.n_classes * opt.n_images, shuffle=False,
                                          sampler=Sampler, drop_last=True, num_workers=8)
# dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.n_classes*opt.n_images, shuffle=True,num_workers=4)

dataset_sizes = len(image_datasets)

use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# --------------------------------------------------------------------

y_loss = []  # loss history


def train_model(model, criterion_part, criterion_tri, criterion_center, criterion_focal,optimizer, scheduler, num_epochs=60):
    # train for each epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)

        ## Each epoch just has a training  phase
        # scheduler.step()
        # warm up

        if epoch == 0:
            c=0
            init_lr=[]
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10.0
                init_lr.append(param_group['lr'])
                c+=1
        elif epoch < 10:
            c=0
            for param_group in optimizer.param_groups:
                param_group['lr'] += init_lr[c]
                c+=1
        elif epoch == 40 or epoch == 60  or epoch == 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10.0

        model.train(True)  # Set model to training mode

        running_loss_tri = 0.0
        running_loss_part = 0.0
        running_loss_center = 0.0
        running_loss_eig = 0.0
        running_loss_focal = 0.0
        running_loss_glo = 0.0

        # for each batch
        for data in dataloaders:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward and get feature
            # feature are of size
            logits_list, glo_feat,glo_fc = model(inputs)


            center_loss = 0.0002 * criterion_center(glo_feat, labels)
            glo_loss = criterion_part(glo_fc, labels)


            # L2 normalization of feature
            # normalize each row, feature is of size batch_size*feature_dim
            glo_feat = 1. * glo_feat / (torch.norm(glo_feat, 2, 1, keepdim=True).expand_as(glo_feat) + 1e-12)

            # compute triplet los
            tri_loss = criterion_tri(glo_feat, labels)
            # tri_loss = torch.FloatTensor(np.array([0]))
            # compute part loss
            part_loss = criterion_part(logits_list[0], labels)
            focal_loss = criterion_focal(logits_list[0], labels)
            for i in range(1, len(logits_list), 1):
                part_loss = part_loss + criterion_part(logits_list[i], labels)
                focal_loss =focal_loss+ criterion_focal(logits_list[i], labels)

            # part_loss = torch.sum(torch.cat([torch.unsqueeze(criterion_part(logit, labels),0) for logit in logits_list]))

            # backward + optimize

            loss = part_loss + tri_loss + glo_loss

            loss.backward()
            # print(tri_loss.requires_grad,part_loss.requires_grad)
            optimizer.step()

            # statistics
            running_loss_tri += tri_loss.data.item()
            running_loss_part += part_loss.data.item()
            running_loss_center += center_loss.data.item()
            running_loss_focal += focal_loss.data.item()
            running_loss_glo += glo_loss.data.item()



        epoch_loss_tri = running_loss_tri / dataset_sizes
        epoch_loss_part = running_loss_part / dataset_sizes
        epoch_loss_center = running_loss_center / dataset_sizes
        epoch_loss_eig = running_loss_eig / dataset_sizes
        epoch_loss_focal = running_loss_focal / dataset_sizes
        epoch_loss_glo = running_loss_glo / dataset_sizes

        print('TriLoss: {:.4f}'.format(epoch_loss_tri))
        print('PartLoss: {:.4f}'.format(epoch_loss_part))
        print('CenterLoss: {:.4f}'.format(epoch_loss_center))
        print('EigLoss: {:.4f}'.format(epoch_loss_eig))
        print('FocalLoss: {:.4f}'.format(epoch_loss_focal))
        print('GloLoss: {:.4f}'.format(epoch_loss_glo))
        with open(os.path.join(opt.outf,'record.txt') , 'a') as acc_file:
            acc_file.write('Epoch: %2d,  TriLoss: %.8f,  PartLoss: %.8f\n' % (epoch, epoch_loss_tri, epoch_loss_part))

        # save model
        if epoch > 20 and epoch % 10 == 9:
            save_network(model, epoch)
        print()

    return model


# Save modeL
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(opt.outf,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


# setting of train phase
model = PCB(class_num=4768)
# load pretrained para without classifier




if use_gpu:
    model = model.cuda()

# set the criterion
triplet_selector = SemihardNegativeTripletSelector(opt.margin)
criterion_tri = OnlineTripletLoss(opt.margin, triplet_selector)

criterion_part = nn.CrossEntropyLoss()
# criterion_part=CrossEntropyLabelSmooth(4768)
criterion_center = CenterLoss(4768)
criterion_focal=FocalLoss(gamma=2)

# updating rule for parameter
ignored_params = list(map(id, model.model.fc.parameters()))
ignored_params += (list(map(id, model.classifier0.parameters()))
                   + list(map(id, model.classifier1.parameters()))
                   + list(map(id, model.classifier2.parameters()))
                   + list(map(id, model.classifier3.parameters()))
                   + list(map(id, model.classifier4.parameters()))
                   + list(map(id, model.classifier5.parameters()))
                   +list(map(id, model.classifier6.parameters() ))
                   # +list(map(id, model.classifier7.parameters() ))
                   )
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': opt.lr/10},
    {'params': model.model.fc.parameters(), 'lr': opt.lr},
    {'params': model.classifier0.parameters(), 'lr': opt.lr},
    {'params': model.classifier1.parameters(), 'lr': opt.lr},
    {'params': model.classifier2.parameters(), 'lr': opt.lr},
    {'params': model.classifier3.parameters(), 'lr': opt.lr},
    {'params': model.classifier4.parameters(), 'lr': opt.lr},
    {'params': model.classifier5.parameters(), 'lr': opt.lr},
    {'params': model.classifier6.parameters(), 'lr': opt.lr},
    # {'params': model.classifier7.parameters(), 'lr': opt.lr}
], weight_decay=5e-4, momentum=0.9, nesterov=True)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
#pretrained_dict = torch.load('')
#model_dict = model.state_dict()
#pretrained_dict = {k: v for k, v in pretrained_dict.items() if k[0:10] != 'classifier'}
#model_dict.update(pretrained_dict)
#model.load_state_dict(model_dict)
# Decay LR by a factor of 0.1 every 20 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
# record

import glob
import shutil
#import os
#import datetime
time1=datetime.datetime.now()
time1_str=datetime.datetime.strftime(time1,'%Y-%m-%d_%H:%M:%S')
script_set1=glob.glob('./*.py')
script_set2=glob.glob('./utils/*.py')+glob.glob('./model/*.py')
saved_folder=opt.outf
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
# Train
model = train_model(model, criterion_part, criterion_tri, criterion_center, criterion_focal,optimizer_ft, exp_lr_scheduler,
                    num_epochs=100)


