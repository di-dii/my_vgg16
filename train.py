import torch
from modle import VGG16_ty
from torch import nn
from torchvision import datasets
from torchvision import transforms

BATCH_SIZE=16
LEARN_RATE=0.01
EPOCH=100
SAVE_STEP=10

#加载数据  

transform=transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
#traindata=datasets.ImageFolder(xxxx)
trainset=datasets.CIFAR10(root='./data',train=True,
                        download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(dataset=trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


vgg16=VGG16_ty(10)  #.cunda()

# loss optimizer scheduler
cost=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(vgg16.parameters(),lr=LEARN_RATE)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

#开始训练
for epoch in range(EPOCH):

    avg_loss=0
    cnt=0
    for images,labels in trainloader:
        #images=images.cuda()
        #labels=labels.cuda()

        #forward + backward + optimize
        optimizer.zero_grad()
        outputs=vgg16(images)
        loss=cost(outputs,labels)
        avg_loss+=loss.data
        cnt+=1
        print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
        loss.backward()
        optimizer.step()
    scheduler.step(avg_loss)

    #保存模型
    if(epoch%SAVE_STEP==0):
        torch.save(vgg16.state_dict(),'vgg16_checkpoint_epoch{}.pt'.format(epoch)) 
    




