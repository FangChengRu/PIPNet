import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time

# net_stride output_size
# 128        2x2
# 64         4x4
# 32         8x8
# pip regression, resnet18, for GSSL
class Pip_resnet18(nn.Module):
    def __init__(self, resnet, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_resnet18, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.sigmoid = nn.Sigmoid()
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.my_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.cls_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(512, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(512, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

        # init
        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        cls1 = self.cls_layer(x)
        offset_x = self.x_layer(x)
        offset_y = self.y_layer(x)
        nb_x = self.nb_x_layer(x)
        nb_y = self.nb_y_layer(x)
        x = self.my_maxpool(x)
        cls2 = self.cls_layer(x)
        x = self.my_maxpool(x)
        cls3 = self.cls_layer(x)
        return cls1, cls2, cls3, offset_x, offset_y, nb_x, nb_y


class Pip_effnetv2(nn.Module):
    def __init__(self, effnet, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_effnetv2, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride

        self.stem_conv = effnet.stem_conv
        self.stem_bn = effnet.stem_bn
        self.stem_act = effnet.stem_act
        self.layer1 = effnet.blocks[:6]
        self.layer2 = effnet.blocks[6:10]
        self.layer3 = effnet.blocks[10:25]
        self.layer4 = effnet.blocks[25:]
        self.head_conv = effnet.head_conv
        self.head_bn = effnet.head_bn
        self.head_act = effnet.head_act

        self.my_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.cls_layer = nn.Conv2d(1280, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(1280, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(1280, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(1280, num_nb * num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(1280, num_nb * num_lms, kernel_size=1, stride=1, padding=0)

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)
        for layer in self.layer4:
            x = layer(x)
        x = self.head_act(self.head_bn(self.head_conv(x)))

        '''x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5'''

        cls1 = self.cls_layer(x)
        offset_x = self.x_layer(x)
        offset_y = self.y_layer(x)
        nb_x = self.nb_x_layer(x)
        nb_y = self.nb_y_layer(x)
        x = self.my_maxpool(x)
        cls2 = self.cls_layer(x)
        x = self.my_maxpool(x)
        cls3 = self.cls_layer(x)
        return cls1, cls2, cls3, offset_x, offset_y, nb_x, nb_y


class Pip_mbnetv3(nn.Module):
    def __init__(self, mbnet, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_mbnetv3, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.features = mbnet.features
        self.conv = mbnet.conv
        self.sigmoid = nn.Sigmoid()

        self.my_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.cls_layer = nn.Conv2d(960, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(960, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(960, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(960, num_nb * num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(960, num_nb * num_lms, kernel_size=1, stride=1, padding=0)

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)

        '''x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5'''

        cls1 = self.cls_layer(x)
        offset_x = self.x_layer(x)
        offset_y = self.y_layer(x)
        nb_x = self.nb_x_layer(x)
        nb_y = self.nb_y_layer(x)
        x = self.my_maxpool(x)
        cls2 = self.cls_layer(x)
        x = self.my_maxpool(x)
        cls3 = self.cls_layer(x)
        return cls1, cls2, cls3, offset_x, offset_y, nb_x, nb_y


if __name__ == '__main__':
    pass
    
