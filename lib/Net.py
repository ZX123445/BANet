from lib.EfficientNet import EfficientNet
from torch import nn
import torch
from thop import profile
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):  #x[2, 512, 24, 24]
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x



class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.context_encoder = EfficientNet.from_pretrained('efficientnet-b5')


        self.reduce4 = BasicConv2d(512, 64, kernel_size=1)
        self.reduce3 = BasicConv2d(176, 64, kernel_size=1)
        self.reduce2 = BasicConv2d(64, 64, kernel_size=1)
        self.reduce1 = BasicConv2d(40, 64, kernel_size=1)


        self.predictor_b = nn.Conv2d(64, 1, 3, padding=1)
        self.predictor_gt3 = nn.Conv2d(64, 1, 3, padding=1)
        self.predictor_gt2 = nn.Conv2d(64, 1, 3, padding=1)
        self.predictor_gt1 = nn.Conv2d(64, 1, 3, padding=1)
        self.predictor_gt0 = nn.Conv2d(64, 1, 3, padding=1)


    def forward(self, x):
        image_shape = x.size()[2:]
        # backbone
        endpoints = self.context_encoder.extract_endpoints(x)
        r1 = endpoints['reduction_2']  #r1[2, 40, 96, 96]
        r2 = endpoints['reduction_3']  #r2[2, 64, 48, 48]
        r3 = endpoints['reduction_4']  #r3[2, 176, 24, 24]
        r4 = endpoints['reduction_5']  #r4[2, 512, 12, 12]

        x1 = self.reduce1(r1)   #r1[2, 64, 96, 96]
        x2 = self.reduce2(r2)   #r2[2, 64, 48, 48]
        x3 = self.reduce3(r3)   #r3[2, 64, 24, 24]
        x4 = self.reduce4(r4)   #r4[2, 64, 12, 12]

        x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        b = x4 + x3


        b = self.predictor_b(b)
        gt3 = self.predictor_gt3(x4)  # gt3[2, 1, 44, 44]
        gt2 = self.predictor_gt2(x3)  # gt2[2, 1, 44, 44]
        gt1 = self.predictor_gt1(x2)  # gt1[2, 1, 44, 44]
        gt0 = self.predictor_gt0(x1)  # gt0[2, 1, 44, 44]


        b = F.interpolate(b, size=image_shape, mode='bilinear', align_corners=True)
        gt3 = F.interpolate(gt3, size=image_shape, mode='bilinear', align_corners=True)
        gt2 = F.interpolate(gt2, size=image_shape, mode='bilinear', align_corners=True)
        gt1 = F.interpolate(gt1, size=image_shape, mode='bilinear', align_corners=True)
        gt0 = F.interpolate(gt0, size=image_shape, mode='bilinear', align_corners=True)

        return b, gt3, gt2, gt1, gt0



if __name__ =='__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    net = Network().cuda()
    data = torch.randn(1, 3, 352, 352).cuda()
    flops, params = profile(net, (data,))
    print('flops: %.2f G, params: %.2f M' % (flops / (1024*1024*1024), params / (1024*1024)))