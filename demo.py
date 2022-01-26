import torch
from torch import nn
from d2l import torch as d2l
from collections import OrderedDict

#这是alexnet
alex1 = nn.Sequential(OrderedDict([
    ('conv1',nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1)),
    ('relu7',nn.ReLU()),
    ('maxpool1',nn.MaxPool2d(kernel_size=3, stride=2)),
    ('conv2',nn.Conv2d(96, 256, kernel_size=5, padding=2)),
    ('relu1',nn.ReLU()),
    ('maxpool2',nn.MaxPool2d(kernel_size=3, stride=2)),
    ('conv3',nn.Conv2d(256, 384, kernel_size=3, padding=1)),
    ('relu2',nn.ReLU()),
    ('conv4',nn.Conv2d(384, 384, kernel_size=3, padding=1)), 
    ('relu3',nn.ReLU()),
    ('conv5',nn.Conv2d(384, 256, kernel_size=3, padding=1)), 
    ('relu4',nn.ReLU()),
    ('maxpool3',nn.MaxPool2d(kernel_size=3, stride=2)),
    ('flatten',nn.Flatten()),
    ('linear1',nn.Linear(6400, 4096)), 
    ('relu5',nn.ReLU()),
    ('drop1',nn.Dropout(p=0.5)),
    ('linear2',nn.Linear(4096, 4096)), 
    ('relu6',nn.ReLU()),
    ('drop2',nn.Dropout(p=0.5)),
    ('linear3',nn.Linear(4096, 10))]))
    
#保存alex1的参数
torch.save(alex1.state_dict(),'alex')

#创建与alex1同样的网络alex2
alex2 = nn.Sequential(OrderedDict([
    ('conv1',nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1)),
    ('relu7',nn.ReLU()),
    ('maxpool1',nn.MaxPool2d(kernel_size=3, stride=2)),
    ('conv2',nn.Conv2d(96, 256, kernel_size=5, padding=2)),
    ('relu1',nn.ReLU()),
    ('maxpool2',nn.MaxPool2d(kernel_size=3, stride=2)),
    ('conv3',nn.Conv2d(256, 384, kernel_size=3, padding=1)),
    ('relu2',nn.ReLU()),
    ('conv4',nn.Conv2d(384, 384, kernel_size=3, padding=1)), 
    ('relu3',nn.ReLU()),
    ('conv5',nn.Conv2d(384, 256, kernel_size=3, padding=1)), 
    ('relu4',nn.ReLU()),
    ('maxpool3',nn.MaxPool2d(kernel_size=3, stride=2)),
    ('flatten',nn.Flatten()),
    ('linear1',nn.Linear(6400, 4096)), 
    ('relu5',nn.ReLU()),
    ('drop1',nn.Dropout(p=0.5)),
    ('linear2',nn.Linear(4096, 4096)), 
    ('relu6',nn.ReLU()),
    ('drop2',nn.Dropout(p=0.5)),
    ('linear3',nn.Linear(4096, 10))]))
    
#加载alex1的参数到alex2中
alex2.load_state_dict(torch.load('alex'))

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

def training(net,train_iter,epochs,lr):
    device=  'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        train_l=0
        for i,(x,y) in enumerate(train_iter):
            optimizer.zero_grad()
            x,y=x.to(device),y.to(device)
            y_hat=net(x)
            l=loss(y_hat,y)
            with torch.no_grad():
                train_l+=l
            l.backward()
            optimizer.step()
        print(f'epoch:{epoch+1}')
        print(f'loss:{train_l/x.shape[0]}')
        print(f'acc:{gpu_accuracy(net,train_iter)}')
        
def gpu_accuracy(net,data_iter):
    if isinstance(net, nn.Module):
        net.eval()  #计算精度是进行推理,要改成eval()模式
    device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    net.to(device)  #模型放入device中
    correct_num=0 #预测正确个数
    all_num=0    #标签元素的个数
    with torch.no_grad():
        for x,y in data_iter:
            x=x.to(device)
            y=y.to(device)   #数据放入gpu
            #得到一个[n][10]的二维数组,返回一个n维数组,每个元素是最大值的下标
            y_hat=net(x).argmax(axis=1)
            #得到每个预测是否相同 ,可能要变成y_hat.type(y.dtype)
            cmp= y_hat==y
            correct_num+=cmp.sum()
            all_num+=y.numel()
        print(f'正确样本数为:{correct_num},样本总数为:{all_num}')
        return  correct_num/all_num
        
num_epochs, lr=10,0.01
training(alex1,train_iter,num_epochs,lr)
training(alex2,train_iter,num_epochs,lr)
