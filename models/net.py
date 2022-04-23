import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable


def conv_bn(inp, oup, stride=1, leaky=0):  # 输入3输出8
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)  # negativa_slope：是控制x为负数时斜率角度。这里为0
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),  # 这里深度卷积原理体现在首先用输入是相同维度进行接收并输出也是接收维度，再用1x1提高维度输出
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)  # inplace表示是否进行原地操作。
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    # if stride==2:
    #     pass

    return nn.Sequential(

        # 修改网络，先升维再提取特征
        # nn.Conv2d(inp,2*inp,1,1,0,bias=False),
        # nn.BatchNorm2d(2*inp),
        # nn.LeakyReLU(negative_slope=leaky,inplace=True),

        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        # nn.Conv2d(inp*2, inp*2, 3, stride, 1, groups=inp, bias=False),
        # nn.BatchNorm2d(inp*2),

        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        # nn.Conv2d(inp*2, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)  # 3个3x3代表一个7x7
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])  # 80x80
        output2 = self.output2(input[1])  # 40x40
        output3 = self.output3(input[2])  # 20x20

        # up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        # output2 = output2 + up3
        # output2 = self.merge2(output2)  # 懂了，这里是将两者和的数据值组合的一幅新图像送输入到conv_bn类型的卷积中，而通道已经在上面的merge函数中定义好了
        # # output2 是跟out1merge之后的
        # # up33=F.interpolate(output3,size=[output1.size(2),output1.size(3)],mode='nearest')
        # up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        # # output1 = output1 + up2
        # # output1 =output1 + up2 + up33
        # output1 = self.merge1(output1)
        # # up1这里，我添加一个跳跃连接，让up1不仅使用up2，也直接使用up3的上采样

        # # 保留原始output2，再上采样一次。
        # up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        # # output2_new = output2 + up3
        # output2 = output2 + up3
        # # output2_new = self.merge2(output2_new) # 将这个用于output2的输出
        # output2 = self.merge2(output2)
        # up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        # up22 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        #
        # # output1 = output1 + up2 + up22
        # output1 = output1 + up2
        #
        # out = [output1, output2, output3]
        # # out = [output1, output2_new, output3]

        # 原网络
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]


        # 自上而下与自下而上结合
        # up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")



        return out


def conv_dw2(inp, oup, stride, leaky=0.1):
    conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

    conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
    conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
    conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        # self.stage2_1=nn.Sequential(conv_dw(64, 128, 2))
        # self.stage2_2=nn.Sequential(conv_dw(128, 128, 1))
        # self.stage2_3=nn.Sequential(conv_dw(128, 128, 1))
        # self.stage2_4=nn.Sequential(conv_dw(128, 128, 1))
        # self.stage2_5=nn.Sequential(conv_dw(128, 128, 1))
        # self.stage2=nn.Sequential(conv_dw(128, 128, 1))

        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)
        self.relu=nn.ReLU()
        # self.cov2=nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)

    def forward(self, x):
        x = self.stage1(x)
        # x_ = self.cov2(x)
        x = self.stage2(x)
        # x = x + x_
        # x = self.relu(x)


        # x =self.stage2_1(x)
        # x_1 =self.stage2_2(x)
        # x = self.stage2_2(x_1)
        # x = self.stage2_3(x)
        # x = self.stage2_4(x)
        # x_2=F.relu(x+x_1)
        # x=self.stage2_5(x_2)
        # x=self.stage2_6(x)
        # x=F.relu(x+x_2)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


# 焯，inception没写出来。

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.conv2_a = InceptionA()

    def forword(self, x):
        x = self.conv2_a(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(InceptionA, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        # self.conv5x5=nn.Conv2d()

    def forward(self, x):
        x_1x1 = self.conv1x1(x)
        x_3x3 = self.conv3x3(x)
        x_5x5 = self.conv3x3(x_3x3)
        y = x_1x1 + x_3x3 + x_5x5
        return y


# 设计自己mobile网络。添加resnet ；backbone
class mobilenetv2(nn.Module):
    def __init__(self):
        super(mobilenetv2, self).__init__()
        self.model = InceptionA
        self.stage1 = nn.Sequential(  # 640x640----320x320
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11#  320x320---160x160
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27   # 160x160---80x80
            conv_dw(64, 64, 1),  # 43
        )
        self.stage1_bn = conv_bn(3, 8, 2, leaky=0.1),
        self.stage1_inception = InceptionA(8)

        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        x = self.model

        return x
