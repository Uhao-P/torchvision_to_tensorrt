import re
# -- third party --
import openbayes_serving as serv
import torch
from math import *
import torch.nn as nn
from torch import  optim


'''定义超参数'''
batch_size = 256        # 批的大小
learning_rate = 1e-2    # 学习率
num_epoches = 10        # 遍历训练集的次数


'''定义网络模型'''
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #5
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #8
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #11
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096,num_classes),
            )
        #self.classifier = nn.Linear(512, 10)
 
    def forward(self, x):
        out = self.features(x) 
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out




class Predictor:
    def __init__(self):
        '''创建model实例对象，并检测是否支持使用GPU'''
        model = VGG16()
        use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
        if use_gpu:
            model = model.cuda()
        '''定义loss和optimizer'''
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        self.static_input = torch.randn(1, 3, 32, 32, device='cuda')
        static_target = torch.rand(1, 10, device='cuda')
        # static_target = torch.Tensor([1])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for i in range(3):
                optimizer.zero_grad(set_to_none=True)
                y_pred = model(self.static_input)
                loss = loss_fn(y_pred, static_target)
                loss.backward()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            static_y_pred = model(self.static_input)
            static_loss = loss_fn(static_y_pred, static_target)
            static_loss.backward()
            optimizer.step()
        self.real_inputs = [torch.rand_like(self.static_input) for _ in range(1)]


    @torch.no_grad()
    def predict(self, data):
        serv.emit_event('request.bin', data)
        return self.pspnet(data)

    
    def pspnet(self, data):
        self.static_input.copy_(self.real_inputs[0])

        return {'predictions': []}


if __name__ == '__main__':  # 如果直接执行了 predictor.py，而不是被其他文件 import
    serv.run(Predictor)  # 开始提供服务
