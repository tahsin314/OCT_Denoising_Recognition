import torch
from torch import nn
import timm
from config import *
from model.utils import *
class Resnet(nn.Module):

    def __init__(self, model_name='resnet18d'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1)
        # print(self.backbone)
        self.in_features = self.backbone.fc.in_features
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)

        return x, layer1, layer2, layer3

class Resne_tDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        nonlinearity = nn.ReLU
        self.decode1 = DecoderBlock(1024, 512)
        self.decode2 = DecoderBlock(512, 256)
        self.decode3 = DecoderBlock(256, 64)
        self.decode4 = DecoderBlock(64, 32)
        self.decode5 = DecoderBlock(32, 16)
        self.conv1 = conv_block(1024, 512)
        self.conv2 = conv_block(512, 256)
        self.conv3 = conv_block(128, 64)
        self.Att1 = Attention_block(512, 512, 256)
        self.Att2 = Attention_block(256, 256, 64)
        self.Att3 = Attention_block(64, 64, 32)
        self.Att4 = Attention_block(64, 64, 32)
        self.conv4 = nn.Conv2d(64, 64, 3, 2, 1)
        self.finalconv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(4, 1, 3, padding=1)
    
    def forward(self, x, l1, l2, l3):
        d1 = self.decode1(l3)
        l2 = self.Att1(d1, l2)
        d1 = torch.cat((l2,d1),dim=1)
        d1 = self.conv1(d1)
        d2 = self.decode2(d1)
        l1 = self.Att2(d2, l1)
        d2 = torch.cat((l1,d2),dim=1)
        d2 = self.conv2(d2)
        d3 = self.decode3(d2)
        d3 = self.conv4(d3)
        x = self.Att3(d3, x)
        d3 = torch.cat((x,d3),dim=1)
        d3 = self.conv3(d3)
        d4 = self.decode4(d3)
        d5 = self.decode5(d4)
        out = self.finalconv2(d5)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class resUne_t(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()
        self.resne_t = Resnet(model_name=encoder_model)
        self.decoder = Resne_tDecoder()
        
    def forward(self, x):
        x, l1, l2, l3 = self.resne_t(x)
        out = self.decoder(x, l1, l2, l3)
        return out