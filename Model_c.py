from BaseNetwork import BaseNetwork
import ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import inception_v3
# inception_v3(pretrained=False)
from model.senet import se_resnext50_32x4d, se_resnext101_32x4d


class SpatialGate2d(nn.Module):

    def __init__(self, in_channels):
        super(SpatialGate2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cal = self.conv1(x)
        cal = self.sigmoid(cal)
        return cal * x

class ChannelGate2d(nn.Module):

    def __init__(self, channels, reduction=2):
        super(ChannelGate2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class scSqueezeExcitationGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super(scSqueezeExcitationGate, self).__init__()
        self.spatial_gate = SpatialGate2d(channels)
        self.channel_gate = ChannelGate2d(channels, reduction=reduction)

    def  forward(self, x, z=None):
        XsSE = self.spatial_gate(x)
        XcSe = self.channel_gate(x)
        return XsSE + XcSe


class UNetResNet34_SE(BaseNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, **kwargs):
        super(UNetResNet34_SE, self).__init__(**kwargs)
        self.resnet = ResNet.resnet34(pretrained=pretrained, SE=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 64


        # self.encoder1 = nn.Sequential(self.resnet.layer1, scSqueezeExcitationGate(64))# 256
        # self.encoder2 = nn.Sequential(self.resnet.layer2, scSqueezeExcitationGate(128))# 512
        # self.encoder3 = nn.Sequential(self.resnet.layer3, scSqueezeExcitationGate(256))# 1024
        # self.encoder4 = nn.Sequential(self.resnet.layer4, scSqueezeExcitationGate(512))# 2048

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.avg_pool = nn.AdaptiveAvgPool2d([1,1])
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512, 256)
        self.last_linear = nn.Linear(256, 28)

    def forward(self, x):
        # batch_size,C,H,W = x.shape

        x = self.conv1(x) # 128

        e1 = self.encoder1(x)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        avg = self.avg_pool(e4)
        flat = avg.view(avg.size(0),-1)
        fc = self.linear(flat)
        fc = self.dropout(fc)
        logit = self.last_linear(fc)
        return logit

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm1', nn.BatchNorm2d(input_num, momentum=0.0003)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', nn.BatchNorm2d(num1, momentum=0.0003)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature

class model50A_DeepSupervion(BaseNetwork):
    def __init__(self, num_classes=28, pretrained=True, **kwargs):
        super(model50A_DeepSupervion, self).__init__(**kwargs)
        if pretrained:
            pretrained = 'imagenet'

        self.num_classes = num_classes
        # self.encoder = se_resnext_50()
        self.encoder = se_resnext50_32x4d(num_classes=1000, pretrained=pretrained)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.dropout = nn.Dropout(0.2)
        # self.linear = nn.Linear(2048, 512)
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        conv1 = self.conv1x1(x)
        conv2 = self.conv2(conv1)  # 1/4
        conv3 = self.conv3(conv2)  # 1/8
        conv4 = self.conv4(conv3)  # 1/16
        conv5 = self.conv5(conv4)  # 1/32

        pool = self.avg_pool(conv5)
        flat = pool.view(pool.size(0), -1)
        # linear = self.linear(flat)
        dropot = self.dropout(flat)
        out = self.last_linear(dropot)
        return out

class model50A_RFClass(BaseNetwork):
    def __init__(self, num_classes=28, pretrained=True, **kwargs):
        super(model50A_RFClass, self).__init__(**kwargs)
        if pretrained:
            pretrained = 'imagenet'

        self.num_classes = num_classes
        #self.encoder = se_resnext_50()
        self.encoder = se_resnext50_32x4d(num_classes=1000, pretrained=pretrained)

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        # self.conv1x1 = nn.Sequential(
        #     nn.Conv2d(3584, 1024, kernel_size=1, padding=0, stride=1),
        #     nn.BatchNorm2d(1024)
        # )

        self.BasicRFB_a = BasicRFB_a(1024, 512)
        self.BasicRFB_b = BasicRFB(2048, 512)
        self.BasicRFB_b1 = BasicRFB(512, 512, 2)
        self.BasicRFB_b2 = BasicRFB(512, 512, 2)

        self.avg_pool1 = nn.AdaptiveAvgPool2d([1, 1])
        self.avg_pool2 = nn.AdaptiveAvgPool2d([1, 1])
        self.avg_pool3 = nn.AdaptiveAvgPool2d([1, 1])
        self.avg_pool4 = nn.AdaptiveAvgPool2d([1, 1])

        self.dropout = nn.Dropout(0.2)
        # self.linear = nn.Linear(1024, 256)
        self.last_linear = nn.Linear(3584, num_classes)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        RFa = self.BasicRFB_a(conv4)
        RFb = self.BasicRFB_b(conv5)
        RFb1 = self.BasicRFB_b1(RFb)

        pool1 = self.avg_pool1(RFa)
        pool2 = self.avg_pool2(RFb)
        pool3 = self.avg_pool3(conv5)
        pool4 = self.avg_pool4(RFb1)

        pool = torch.cat((pool1, pool2, pool3, pool4), 1)
        # pool_c = self.conv1x1(pool)
        flat = pool.view(pool.size(0),-1)
        # linear = self.linear(flat)
        dropot = self.dropout(flat)
        out = self.last_linear(dropot)
        return out

class model50A_DenseASPP(BaseNetwork):
    def __init__(self, num_classes=28, pretrained=False, **kwargs):
        super(model50A_DenseASPP, self).__init__(**kwargs)
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = None

        num_features = 2048
        d_feature0 = 128
        d_feature1 = 64
        dropout0 = 0.2
        self.num_classes = num_classes
        #self.encoder = se_resnext_50()
        self.encoder = se_resnext50_32x4d(num_classes=1000, pretrained=pretrained)

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=True)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)

        self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])

        num_features = num_features + 5 * d_feature1

        self.classification = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels=num_features, out_channels=4, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=16, mode='bilinear')
        )

        self.dropout = nn.Dropout(0.2)
        # self.linear = nn.Linear(1024, 256)
        self.last_linear = nn.Linear(2368, num_classes)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        feature = conv5
        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        pool = self.avg_pool(feature)
        flat = pool.view(pool.size(0),-1)
        # linear = self.linear(flat)
        dropot = self.dropout(flat)
        out = self.last_linear(dropot)

        img_out = self.classification(feature)
        return out, img_out

class model101A_DenseASPP(BaseNetwork):
    def __init__(self, num_classes=28, pretrained=False, **kwargs):
        super(model101A_DenseASPP, self).__init__(**kwargs)
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = None

        num_features = 2048
        d_feature0 = 128
        d_feature1 = 64
        dropout0 = 0.2
        self.num_classes = num_classes
        #self.encoder = se_resnext_50()
        self.encoder = se_resnext101_32x4d(num_classes=1000, pretrained=pretrained)

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=True)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)

        self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])

        num_features = num_features + 5 * d_feature1

        self.classification = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels=num_features, out_channels=4, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=16, mode='bilinear')
        )

        self.dropout = nn.Dropout(0.2)
        # self.linear = nn.Linear(1024, 256)
        self.last_linear = nn.Linear(2048, num_classes)
    def forward(self, x):
        conv1 = self.conv1(x)
        p = F.max_pool2d(conv1, kernel_size=2, stride=2)
        conv2 = self.conv2(p) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        # feature = conv5
        # aspp3 = self.ASPP_3(feature)
        # feature = torch.cat((aspp3, feature), dim=1)
        #
        # aspp6 = self.ASPP_6(feature)
        # feature = torch.cat((aspp6, feature), dim=1)
        #
        # aspp12 = self.ASPP_12(feature)
        # feature = torch.cat((aspp12, feature), dim=1)
        #
        # aspp18 = self.ASPP_18(feature)
        # feature = torch.cat((aspp18, feature), dim=1)
        #
        # aspp24 = self.ASPP_24(feature)
        # feature = torch.cat((aspp24, feature), dim=1)

        pool = self.avg_pool(conv5)
        flat = pool.view(pool.size(0),-1)
        # linear = self.linear(flat)
        dropot = self.dropout(flat)
        out = self.last_linear(dropot)

        # img_out = self.classification(feature)
        # return out, img_out
        return out