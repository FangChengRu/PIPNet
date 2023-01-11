import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

# net_stride output_size
# 128        2x2
# 64         4x4
# 32         8x8
# pip regression, resnet101
class Pip_resnet101(nn.Module):
    def __init__(self, resnet, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_resnet101, self).__init__()
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
        if self.net_stride == 128:
            self.layer5 = nn.Conv2d(2048, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.layer6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn6 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)

            nn.init.normal_(self.layer6.weight, std=0.001)
            if self.layer6.bias is not None:
                nn.init.constant_(self.layer6.bias, 0)
            nn.init.constant_(self.bn6.weight, 1)
            nn.init.constant_(self.bn6.bias, 0)
        elif self.net_stride == 64:
            self.layer5 = nn.Conv2d(2048, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)
        elif self.net_stride == 32:
            pass
        else:
            print('No such net_stride!')
            exit(0)

        self.cls_layer = nn.Conv2d(2048, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(2048, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(2048, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(2048, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(2048, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

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
        if self.net_stride == 128:
            x = F.relu(self.bn5(self.layer5(x)))
            x = F.relu(self.bn6(self.layer6(x)))
        elif self.net_stride == 64:
            x = F.relu(self.bn5(self.layer5(x)))
        else:
            pass
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5

# net_stride output_size
# 128        2x2
# 64         4x4
# 32         8x8
# pip regression, resnet50
class Pip_resnet50(nn.Module):
    def __init__(self, resnet, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_resnet50, self).__init__()
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
        if self.net_stride == 128:
            self.layer5 = nn.Conv2d(2048, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.layer6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn6 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)

            nn.init.normal_(self.layer6.weight, std=0.001)
            if self.layer6.bias is not None:
                nn.init.constant_(self.layer6.bias, 0)
            nn.init.constant_(self.bn6.weight, 1)
            nn.init.constant_(self.bn6.bias, 0)
        elif self.net_stride == 64:
            self.layer5 = nn.Conv2d(2048, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)
        elif self.net_stride == 32:
            pass
        else:
            print('No such net_stride!')
            exit(0)

        self.cls_layer = nn.Conv2d(2048, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(2048, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(2048, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(2048, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(2048, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

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
        if self.net_stride == 128:
            x = F.relu(self.bn5(self.layer5(x)))
            x = F.relu(self.bn6(self.layer6(x)))
        elif self.net_stride == 64:
            x = F.relu(self.bn5(self.layer5(x)))
        else:
            pass
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5

# net_stride output_size
# 128        2x2
# 64         4x4
# 32         8x8
# pip regression, resnet18
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
        if self.net_stride == 128:
            self.layer5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.layer6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn6 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)

            nn.init.normal_(self.layer6.weight, std=0.001)
            if self.layer6.bias is not None:
                nn.init.constant_(self.layer6.bias, 0)
            nn.init.constant_(self.bn6.weight, 1)
            nn.init.constant_(self.bn6.bias, 0)
        elif self.net_stride == 64:
            self.layer5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)
        elif self.net_stride == 32:
            pass
        elif self.net_stride == 16:
            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_deconv1 = nn.BatchNorm2d(512)
            nn.init.normal_(self.deconv1.weight, std=0.001)
            if self.deconv1.bias is not None:
                nn.init.constant_(self.deconv1.bias, 0)
            nn.init.constant_(self.bn_deconv1.weight, 1)
            nn.init.constant_(self.bn_deconv1.bias, 0)
        else:
            print('No such net_stride!')
            exit(0)

        self.cls_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(512, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(512, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

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
        if self.net_stride == 128:
            x = F.relu(self.bn5(self.layer5(x)))
            x = F.relu(self.bn6(self.layer6(x)))
        elif self.net_stride == 64:
            x = F.relu(self.bn5(self.layer5(x)))
        elif self.net_stride == 16:
            x = F.relu(self.bn_deconv1(self.deconv1(x)))
        else:
            pass
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5

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

        self.cls_layer = nn.Conv2d(1280, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(1280, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(1280, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(1280, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(1280, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

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
        
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5

class Pip_mbnetv2(nn.Module):
    def __init__(self, mbnet, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_mbnetv2, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.features = mbnet.features
        self.sigmoid = nn.Sigmoid()

        self.cls_layer = nn.Conv2d(1280, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(1280, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(1280, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(1280, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(1280, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

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
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5

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

        self.cls_layer = nn.Conv2d(960, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(960, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(960, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(960, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(960, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

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
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5

class Pip_mbvitv2(nn.Module):
    def __init__(self, mbvit, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_mbvitv2, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.stem = mbvit.stem
        self.stages = mbvit.stages
        # self.sigmoid = nn.Sigmoid()

        self.cls_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(512, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(512, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

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
        x = self.stem(x)
        x = self.stages(x)

        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        
        return x1, x2, x3, x4, x5

 
def conv_bn(inp, oup, stride, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def depth_conv2d(inp, oup, kernel=1, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size = kernel, stride = stride, padding=pad, groups=inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1)
    )


def conv_dw(inp, oup, stride, kernel_size=5, padding=2):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup)
    )


def conv_pw(inp, oup, stride, kernel_size=5, padding=2):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup)
    )


class BlazeBlock(nn.Module):
    def __init__(self, inp, oup, double_oup=None, stride=1, kernel_size=5):
        super(BlazeBlock, self).__init__()
        assert stride in [1, 2] 
        self.stride = stride
        self.inp = inp 
        self.use_pooling = self.stride != 1 or double_oup != None
        self.shortcut_oup = double_oup or oup
        self.actvation = nn.ReLU(inplace=True)

        if double_oup == None: 
            
            self.conv = nn.Sequential( 
                    conv_dw(inp, oup, stride, kernel_size)
                )
        else:
            self.conv = nn.Sequential(
                    conv_dw(inp, oup, stride, kernel_size),
                    nn.ReLU(inplace=True),
                    conv_pw(oup, double_oup, 1, kernel_size),
                    nn.ReLU(inplace=True)
                )
        
        if self.use_pooling:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=inp, out_channels=self.shortcut_oup, kernel_size=1, stride=1),
                nn.BatchNorm2d(self.shortcut_oup),
                nn.ReLU(inplace=True)
            ) 


    def forward(self,x):

        h = self.conv(x)

        if self.use_pooling:
            x = self.shortcut(x)

        z = h + x
        # print(z.size())
        return self.actvation(h + x)
        
        
class PIP_Blaze(nn.Module):
    def __init__(self, t_net, num_nb, num_lms=68, input_size=256, net_stride=32, init=False, ts=False):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(PIP_Blaze, self).__init__()
        self.num_classes = 2
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.t_net = t_net
        self.ts = ts
        if self.t_net == 'resnet50' or self.t_net == 'resnet101':
            self.block_expansion = 4
        else:
            self.block_expansion = 1
        self.inplanes = 64
        self.planes = 64 * self.block_expansion
        self.hid_planes = self.planes // 4

        self.conv1 = conv_bn(3, self.inplanes, stride=2)
        self.conv2 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv3 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv4 = BlazeBlock(self.inplanes, self.planes, stride=2)
        self.inplanes = self.planes
        self.planes = 128 * self.block_expansion
        self.hid_planes = self.planes // 4
        self.conv5 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv6 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv7 = BlazeBlock(self.inplanes, self.hid_planes, self.planes, stride=2)
        self.inplanes = self.planes
        self.planes = 256 * self.block_expansion
        self.hid_planes = self.planes // 4
        self.conv8 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv9 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv10 = BlazeBlock(self.inplanes, self.hid_planes, self.planes, stride=2)
        self.inplanes = self.planes
        self.planes = 512 * self.block_expansion
        self.hid_planes = self.planes // 4
        self.conv11 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv12 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv13 = BlazeBlock(self.inplanes, self.hid_planes, self.planes, stride=2)
        
        self.cls_layer = nn.Conv2d(self.planes, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(self.planes, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(self.planes, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(self.planes, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(self.planes, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        
        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    
    def forward(self,inputs):
        feats = []

        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        feats.append(x4)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        feats.append(x7)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        feats.append(x10)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        feats.append(x13)

        out_cls = self.cls_layer(x13)
        out_x = self.x_layer(x13)
        out_y = self.y_layer(x13)
        out_nb_x = self.nb_x_layer(x13)
        out_nb_y = self.nb_y_layer(x13)
        
        if self.ts:
            return feats, out_cls, out_x, out_y, out_nb_x, out_nb_y
        else:
            return out_cls, out_x, out_y, out_nb_x, out_nb_y


class PIP_Blaze_down1(nn.Module):
    def __init__(self, t_net, num_nb, num_lms=68, input_size=256, net_stride=32, init=False, ts=False):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(PIP_Blaze_down1, self).__init__()
        self.num_classes = 2
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.t_net = t_net
        self.ts = ts
        if self.t_net == 'resnet50' or self.t_net == 'resnet101':
            self.block_expansion = 4
        else:
            self.block_expansion = 1
        self.inplanes = 64
        self.planes = 64 * self.block_expansion
        self.hid_planes = self.planes // 4

        self.conv1 = conv_bn(3, self.inplanes, stride=2)
        self.conv2 = BlazeBlock(self.inplanes, self.inplanes, stride=2)
        self.conv3 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv4 = BlazeBlock(self.inplanes, self.planes)
        self.inplanes = self.planes
        self.planes = 128 * self.block_expansion
        self.hid_planes = self.planes // 4
        self.conv5 = BlazeBlock(self.inplanes, self.inplanes, stride=2)
        self.conv6 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv7 = BlazeBlock(self.inplanes, self.hid_planes, self.planes)
        self.inplanes = self.planes
        self.planes = 256 * self.block_expansion
        self.hid_planes = self.planes // 4
        self.conv8 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes, stride=2)
        self.conv9 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv10 = BlazeBlock(self.inplanes, self.hid_planes, self.planes)
        self.inplanes = self.planes
        self.planes = 512 * self.block_expansion
        self.hid_planes = self.planes // 4
        self.conv11 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes, stride=2)
        self.conv12 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv13 = BlazeBlock(self.inplanes, self.hid_planes, self.planes)
        
        self.cls_layer = nn.Conv2d(self.planes, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(self.planes, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(self.planes, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(self.planes, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(self.planes, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        
        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    
    def forward(self,inputs):
        feats = []

        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        feats.append(x4)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        feats.append(x7)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        feats.append(x10)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        feats.append(x13)

        out_cls = self.cls_layer(x13)
        out_x = self.x_layer(x13)
        out_y = self.y_layer(x13)
        out_nb_x = self.nb_x_layer(x13)
        out_nb_y = self.nb_y_layer(x13)
        
        if self.ts:
            return feats, out_cls, out_x, out_y, out_nb_x, out_nb_y
        else:
            return out_cls, out_x, out_y, out_nb_x, out_nb_y


class PIP_Blaze_small(nn.Module):
    def __init__(self, t_net, num_nb, num_lms=68, input_size=256, net_stride=32, init=False, ts=False):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(PIP_Blaze_small, self).__init__()
        self.num_classes = 2
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.t_net = t_net
        self.ts = ts
        if self.t_net == 'resnet50' or self.t_net == 'resnet101':
            self.block_expansion = 4
        else:
            self.block_expansion = 1
        self.inplanes = 24
        self.planes = 48 * self.block_expansion
        self.hid_planes = self.planes // 4

        self.conv1 = conv_bn(3, self.inplanes, stride=2)
        self.conv2 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv3 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv4 = BlazeBlock(self.inplanes, self.planes, stride=2)
        self.inplanes = self.planes
        self.planes = 96 * self.block_expansion
        self.hid_planes = self.planes // 4
        self.conv5 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv6 = BlazeBlock(self.inplanes, self.inplanes)
        self.conv7 = BlazeBlock(self.inplanes, self.hid_planes, self.planes, stride=2)
        self.inplanes = self.planes
        self.planes = 96 * self.block_expansion
        self.hid_planes = self.planes // 4
        self.conv8 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv9 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv10 = BlazeBlock(self.inplanes, self.hid_planes, self.planes, stride=2)
        self.inplanes = self.planes
        self.planes = 192 * self.block_expansion
        self.hid_planes = self.planes // 4
        self.conv11 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv12 = BlazeBlock(self.inplanes, self.hid_planes, self.inplanes)
        self.conv13 = BlazeBlock(self.inplanes, self.hid_planes, self.planes, stride=2)
        
        self.cls_layer = nn.Conv2d(self.planes, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(self.planes, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(self.planes, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(self.planes, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(self.planes, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        
        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    
    def forward(self,inputs):
        feats = []

        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        feats.append(x4)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        feats.append(x7)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        feats.append(x10)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        feats.append(x13)

        out_cls = self.cls_layer(x13)
        out_x = self.x_layer(x13)
        out_y = self.y_layer(x13)
        out_nb_x = self.nb_x_layer(x13)
        out_nb_y = self.nb_y_layer(x13)
        
        if self.ts:
            return feats, out_cls, out_x, out_y, out_nb_x, out_nb_y
        else:
            return out_cls, out_x, out_y, out_nb_x, out_nb_y

if __name__ == '__main__':
    pass

