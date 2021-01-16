import torch
from modle import VGG16_ty
from torch import nn
from torchvision import datasets
from torchvision import transforms

weight_PATH='./xxx.pt'

testset=datasets.CIFAR10(root='./data',train=True,
                        download=True)
trainloader=torch.utils.data.DataLoader(dataset=trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

vgg16=VGG16_ty(10)   #.cunda()

correct = 0
total = 0

checkpoint=torch.load(weight_PATH)
vgg16.load_state_dict(checkpoint)    #checkpoint['state_dict']

vgg16.eval()

for images, labels in testLoader:
    #images = images.cuda()
    _, outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print(predicted, labels, correct, total)
    print("avg acc: %f" % (100* correct/total))