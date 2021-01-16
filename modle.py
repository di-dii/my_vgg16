''' 定义模型网络结构 '''
from torch import nn

from torchvision import models


#vgggg=models.vgg16()


class VGG16_ty(nn.Module):        #继承nn.Module类
    def __init__(self,num_class=1000):          #构造函数  
        super(VGG16_ty,self).__init__()     #调用父类的构造函数

        net=[]

        #group1
        net.append(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=(1,1))) # output  224*224*64
        net.append(nn.ReLU(inplace=True))  #激活函数  上面是2维卷积 2维指的是kernel的移动轨迹 
        net.append(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)) # out 224*224*64 #pd=1 == pd=(1,1) 
        net.append(nn.ReLU(inplace=True))    # True 表示直接修改输入的数 而不是返回新申请的空间
        net.append(nn.MaxPool2d(kernel_size=2,stride=2))     #out 112*112*64    #最大值池化

        #group2
        net.append(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)) #out 112*112*128
        net.append(nn.ReLU(inplace=True))
        net.append(nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)) #out 112*112*128
        net.append(nn.ReLU(inplace=True))
        net.append(nn.MaxPool2d(kernel_size=2,stride=2))   #out 64*64*128

        #group3
        net.append(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)) #out 56*56*256
        net.append(nn.ReLU(inplace=True)) 
        net.append(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)) #out 56*56*256
        net.append(nn.ReLU(inplace=True)) 
        net.append(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)) #out 56*56*256
        net.append(nn.ReLU(inplace=True)) 
        net.append(nn.MaxPool2d(kernel_size=2,stride=2))   #out 28*28*256

        #group4
        net.append(nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)) #out 28*28*512
        net.append(nn.ReLU(inplace=True)) 
        net.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)) #out 28*28*512
        net.append(nn.ReLU(inplace=True)) 
        net.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)) #out 28*28*512
        net.append(nn.ReLU(inplace=True)) 
        net.append(nn.MaxPool2d(kernel_size=2,stride=2))  #out 14*14*512

        #group4
        net.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)) #out 14*14*512
        net.append(nn.ReLU(inplace=True)) 
        net.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)) #out 14*14*512
        net.append(nn.ReLU(inplace=True)) 
        net.append(nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)) #out 14*14*512
        net.append(nn.ReLU(inplace=True)) 
        net.append(nn.MaxPool2d(kernel_size=2,stride=2))  #out 7*7*512

        self.extract_feature=nn.Sequential(*net)


        category=[]
        category.append(nn.Linear(in_features=7*7*512,out_features=4096))  #out 4096   #线性全连接
        category.append(nn.ReLU(inplace=True))
        category.append(nn.Dropout(p=0.5))
        category.append(nn.Linear(4096,4096))  #out 4096
        category.append(nn.ReLU(inplace=True))
        category.append(nn.Dropout(p=0.5))     
        category.append(nn.Linear(4096,num_class))  #out numclass

        self.classifier=nn.Sequential(*category)


    def forward(self,x):
        x=self.extract_feature(x)
        # 即从[1,512,7,7]变形为[1,512*7*7]
        x=x.view(x.size(0),-1)    #tensor变形  -1表示根据前面的参数a及原来的x自动计算出参数b  及变形为 a*b  这里是x.size(0)*b  
        x=self.classifier(x)
        return x