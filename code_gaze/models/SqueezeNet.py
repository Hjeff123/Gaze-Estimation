import torch
import torch.nn as nn


class Fire(nn.Module):

    def __init__(self,inplanes,squeeze_planes,expand1x1_planes,expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes=inplanes
        self.squeeze=nn.Conv2d(inplanes,squeeze_planes,kernel_size=1)
        self.squeeze_activation=nn.ReLU(inplace=True)
        self.expand1x1=nn.Conv2d(squeeze_planes,expand1x1_planes,kernel_size=1)
        self.expand1x1_activation=nn.ReLU(inplace=True)
        self.expand3x3=nn.Conv2d(squeeze_planes,expand3x3_planes,kernel_size=3,padding=1)
        self.expand3x3_activation=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ],1)

class Model(nn.Module):
    def __init__(self,version=1.1,num_classes=1000):
        super(Model,self).__init__()

        if version not in [1.0,1.1]:
            raise ValueError("Unsupported SqueezeNet Version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes=num_classes
        if version==1.0:
            self.features=nn.Sequential(
                nn.Conv2d(3,96,kernel_size=7,stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=2), #向上取整
                Fire(96,16,64,64),
                Fire(128,16,64,64),
                Fire(128,32,128,128),
                nn.MaxPool2d(kernel_size=3,stride=2),
                Fire(256,32,128,128),
                Fire(256,48,192,192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        self.fc1 = nn.Linear(66, 2)
        self.conv3 = nn.Conv2d(512, 64, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1,padding=1)  # 若卷积核大小为3x3, 那么就应该设定padding=1, 即填充1层边缘像素; 若卷积核大小为7x7, 那么就应该设定padding=3

            #Final convolution is initialized differently form the rest
            # final_conv = nn.Conv2d(512,self.num_classes,kernel_size=1)
            # self.classifier=nn.Sequential(
            #     nn.Dropout(p=0.5),
            #     final_conv,
            #     nn.ReLU(inplace=True),
            #     nn.AdaptiveAvgPool1d(1,1))

            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d):
            #         if m is final_conv:
            #             init.normal_(m.weight, mean=0.0, std=0.01)
            #         else:
            #             init.kaiming_uniform_(m.weight)
            #         if m.bias is not None:
            #             init.constant_(m.bias, 0)




    def initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)



    def forward(self,x,y):
        x=self.features(x)   # -1,512,1,2
       #x=self.classifier(x)
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.nn.functional.adaptive_avg_pool2d(x,(1,1)) # -1,512,1,1
        x=self.conv3(x)
        # attention, added by JZ Chen
        #x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        z = self.conv4(x)
        z = torch.nn.functional.sigmoid(z)
        z = z.repeat(1, 64, 1, 1)
        z = torch.mul(x, z)
        x = torch.add(z, x)
        # attention

        x = x.view(-1,64) # -1,512
        x = torch.cat([x, y], dim=1)  # concated with pose -1,514
        x = self.fc1(x) # -1,2



        #return x.view(x.size(0),self.num_classes)
        return x