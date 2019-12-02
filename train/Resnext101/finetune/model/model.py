import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from model_IBN import *

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=False, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnext101_32x8d(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.avgpool_glo=nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.1)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 512))
        self.inplanes=2048
        self.bnneck=nn.BatchNorm1d(self.inplanes)
        self.classifier6=ClassBlock(2048, class_num, True, False, 512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)#b,2048,24,8
        '''
        x_eig=x.view(x.size(0),x.size(1),-1)
        # eig_value loss

        x_mm = x_eig.transpose(1, 2).bmm(x_eig)  # b,2048,2048
        x_mm_div = torch.sqrt(torch.sum(torch.sum(torch.pow(x_mm, 2), dim=2), dim=1)).unsqueeze(1).unsqueeze(
            2).expand_as(x_mm)
        x_mm = torch.div(x_mm, x_mm_div)
        eig_value = []
        for i in range(x_eig.size(0)):
            e, v = torch.symeig(x_mm[i, :, :], eigenvectors=True)
            eig_value.append(torch.pow(e[-1] - e[0], 2))
        eig_value = torch.mean(torch.stack(eig_value, dim=0))
        '''
        x = self.avgpool(x)
        glo_fea=self.avgpool_glo(x).squeeze(3).squeeze(2)#b,2048
        glo_fea=self.bnneck(glo_fea)


        if not self.training:
            part_fea=[self.classifier0.add_block(x[:,:,0].squeeze()),self.classifier1.add_block(x[:,:,1].squeeze()), \
                      self.classifier2.add_block(x[:,:,2].squeeze()),self.classifier3.add_block(x[:,:,3].squeeze()), \
                      self.classifier4.add_block(x[:,:,4].squeeze()),self.classifier5.add_block(x[:,:,5].squeeze())]
            return glo_fea,part_fea#glo_fea b,2048;part_fea b,512
        glo_fea = self.dropout(glo_fea)
        glo_fc=self.classifier6(glo_fea)

        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i]) 
     
        return y,glo_fea,glo_fc

