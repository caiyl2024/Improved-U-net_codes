import torch
import torch.nn as nn

import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch
bn_mom = 0.0003
#from torchsummary import summary

class ChannelAttention(nn.Module):
    def __init__(self, in_planes,depthwiseout,ratio=16):
        super(ChannelAttention, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)#自适应平均池化，将输入转换为channel*1*1大小
        #self.max_pool = nn.AdaptiveMaxPool2d(1)
        


        self.depthwise = nn.Conv2d(
            in_channels=in_planes,
            out_channels=in_planes,
            kernel_size=depthwiseout,
            stride=1,
            groups=in_planes,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_planes,
            out_channels=in_planes,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=False
        )




        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)#降低维度1*1卷积
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        #print(list(x.size())[2])
        out = self.fc2(self.relu1(self.fc1(self.pointwise(self.depthwise(x)))))
        
        #out = avg_out + max_out
        return self.sigmoid(out)*x


class ChannelAttention2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#自适应平均池化，将输入转换为channel*1*1大小
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)#降低维度1*1卷积
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
    
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x


class Conv(nn.Module):
    def __init__(self,ch_in,ch_out,bn = True):
        super(Conv,self).__init__()
        if bn:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )

    def forward(self,x):
        x = self.Conv(x)
        return x


class Conv1(nn.Module): # 只做一次卷积
    def __init__(self,ch_in,ch_out,bn = True):
        super(Conv1,self).__init__()
        if bn:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )

    def forward(self,x):
        x = self.Conv(x)
        return x


class VGG(nn.Module):  #用来提取5层特征
    def __init__(self, band_num):
        super(VGG, self).__init__()
        self.band_num = band_num

        self.conv1 = Conv(ch_in=band_num,ch_out=64)
        self.ca1 = ChannelAttention2(in_planes=64)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = Conv(ch_in=64,ch_out=128)
        self.ca2 = ChannelAttention2(in_planes=128)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = Conv(ch_in=128, ch_out=256)
        self.conv3_1 = Conv1(ch_in=256, ch_out=256) # 再做一次额外的卷积
        self.ca3 = ChannelAttention(in_planes=256,depthwiseout=128)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = Conv(ch_in=256, ch_out=512)
        self.conv4_1 = Conv1(ch_in=512, ch_out=512)  # 再做一次额外的卷积
        self.ca4 = ChannelAttention(in_planes=512,depthwiseout=64)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = Conv(ch_in=512, ch_out=512)
        self.ca5 = ChannelAttention(in_planes=512,depthwiseout=32)
        self.conv5_1 = Conv1(ch_in=512, ch_out=512)  # 再做一次额外的卷积

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x=x+self.ca1(x)
        feat1 = x
        x = self.maxpool(x)

        x = self.conv2(x)
        x=x+self.ca2(x)
        feat2 = x
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.conv3_1(x)
        x=x+self.ca3(x)
        feat3 = x
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.conv4_1(x)
        x=x+self.ca4(x)
        feat4 = x
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.conv5_1(x)
        x=x+self.ca5(x)
        feat5 = x

        return feat1, feat2, feat3, feat4, feat5
# [[-1, 64, 512, 512], [-1, 128, 256, 256], [-1, 256, 128, 128], [-1, 512, 64, 64], [-1, 512, 32, 32]] # 五个特征的形状

    def _initialize_weights(self): # 这是单个网络的权重初始化 主网络初始化了,单网络应该就不需要了
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class my_Unet(nn.Module):
    def __init__(self,num_class, band_num):
        super(my_Unet, self).__init__()
        self.main_network = VGG(band_num=band_num)  # 提取五层特征
        self.up = nn.Upsample(scale_factor=2)

        self.convP4 = Conv(ch_in=512+512, ch_out=512)
        self.up = nn.Upsample(scale_factor=2)

        self.convP3 = Conv(ch_in=512+256, ch_out=256)
        self.up = nn.Upsample(scale_factor=2)

        self.convP2 = Conv(ch_in=256+128, ch_out=128)
        self.up = nn.Upsample(scale_factor=2)

        self.convP1 = Conv(ch_in=128+64, ch_out=64)



        self.result = nn.Conv2d(64,num_class,kernel_size=1,stride=1,padding=0)  #  [-1, 3, 512, 512] 和下面这种写法是一样的
        #self.conv_fina = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
        #self.softmax = nn.Softmax(dim=0)   # 在这里输出的波段就是类别个数    [-1, 3, 512, 512]

        #self.initialize_weights(my_Unet)
    def forward(self,x):
        feat1, feat2, feat3, feat4, feat5 = self.main_network(x)#按vgg执行到底部，执行完五层，这里是encode过程，decode在下面
        feat5_up = self.up(feat5)

        P4 = torch.cat((feat5_up, feat4), dim=1)  # band : 512 +  512
        P4 = self.convP4(P4) # 这就是两次卷积
        P4_up = self.up(P4)   # 512

        P3 = torch.cat((P4_up, feat3), dim=1)   # 512 + 256
        P3 = self.convP3(P3)
        P3_up = self.up(P3)

        P2 = torch.cat((P3_up, feat2), dim=1)
        P2 = self.convP2(P2)
        P2_up = self.up(P2)

        P1 = torch.cat((P2_up, feat1), dim=1)
        P1 = self.convP1(P1)

        #result = self.conv_fina(P1)
        #result = self.softmax(result)
        result = self.result(P1)

        return result

    def initialize_weights(self, *stages):# 这应该是复合网络下的权重初始化
        print('权重初始化被调用')
        for modules in stages:
            print(modules)
            for module in modules.modules(self):

                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    nn.init.constant_(module.bias, 0)

def main():  #这是查看一个网络结构的方法
    net = my_Unet()
    device = torch.device('cuda:0')
    model = net.to(device)
    #summary(model,(26,512,512))

if __name__ == '__main__':
    main()
