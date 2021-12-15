from copy import deepcopy
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torch import nn
from torch.nn import *
from torch.nn import functional as F
from torchvision import models
from typing import Optional
from .utils import *
from .triplet_attention import *
from .cbam import *
from .botnet import *
from losses.arcface import ArcMarginProduct
import sys
sys.path.append('../pytorch-image-models/pytorch-image-models-master')
import timm
from pprint import pprint

class VarResnet(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', num_class=3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1)
        self.in_features = self.backbone.fc.in_features
        self.meanhead = Head(self.in_features,num_class, activation='mish')
        self.varhead = Head(self.in_features,num_class, activation='mish')
        # self.mean = self.backbone.layer4.copy()
        # self.var = self.backbone.layer4.copy()
        self.out = nn.Linear(self.in_features, num_class)
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # mean = self.mean(x)
        mean = self.meanhead(x)
        # var = self.var(x)
        var = self.varhead(x)
        z = self.reparameterization(mean, var)
        # print(z.size())
        return z