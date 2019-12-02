import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from model.model_IBN  import *

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
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
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
class PCB_ibn(nn.Module):
    def __init__(self, class_num ):
        super(PCB_ibn, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        #model_ft = models.resnet50(pretrained=True)
        #model_ft = models.resnext101_32x8d(pretrained=True)
        model_ft = resnet50_ibn_a(last_stride=1, pretrained=False)
        #model_ft.load_param('/home/guoqiang/reid_competition/checkpoints/ibn81.2.pth')
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.avgpool_glo=nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.1)
        self.bnneck = nn.BatchNorm1d(2048)
        self.classifier6 = ClassBlock(2048, class_num, True, False, 512)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 512))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        glo_fea_tri=self.avgpool_glo(x).squeeze(3).squeeze(2)#b,2048
        glo_fea = self.bnneck(glo_fea_tri)
        glo_fea = self.dropout(glo_fea)
        y_glo = self.classifier6(glo_fea)
        part_fea=[self.classifier0.add_block(x[:,:,0].squeeze()),self.classifier1.add_block(x[:,:,1].squeeze()),self.classifier2.add_block(x[:,:,2].squeeze()),self.classifier3.add_block(x[:,:,3].squeeze()),
                  self.classifier4.add_block(x[:,:,4].squeeze()),self.classifier5.add_block(x[:,:,5].squeeze())]

        if not self.training:
            return glo_fea,part_fea#glo_fea b,2048;part_fea b,512
        #x = self.dropout(x)

        # get six part feature batchsize*2048*6
        predict=[self.classifier0(x[:,:,0].squeeze()),self.classifier1(x[:,:,1].squeeze()),self.classifier2(x[:,:,2].squeeze()),self.classifier3(x[:,:,3].squeeze()),
                  self.classifier4(x[:,:,4].squeeze()),self.classifier5(x[:,:,5].squeeze())]

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y,glo_fea_tri,y_glo

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

"""
# debug model structure
#net = ft_net(751)
net = PCB(751)
net.eval()
#print(net)
input = Variable(torch.FloatTensor(8, 3, 384, 192))
output,b = net(input)
print('net output size:')
print(b[1].size())
"""
