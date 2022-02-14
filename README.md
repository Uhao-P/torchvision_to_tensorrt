# **openbayes-trt**

> openbayes-trt是一个将python常规框架训练好的模型转换为tensort模型并用来进行预测的项目。本项目是在[tensorrt](https://github.com/NVIDIA/TensorRT)与[torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)两个官方项目的基础上进行开发，目前只支持pytorch部分模型转化trt操作。 



## build

### Prerequisites

> 需要先编译安装tensorrt以及一些系统环境，具体安装方法可从下方链接跳转。

+ [tensorrt](https://github.com/NVIDIA/TensorRT) >= 7.0.0

+ pytorch >= 1.6.0

### install openbayes_plugin

```shell
git clone git@github.com:signcl/openbayes-trt.git
cd openbayes-trt/openbayes_plugin
mkdir build && cd build
cmake -DTENSORRT_DIR=${TENSORRT_DIR} ..
make -j10
export OPENBAYES_LIBRARY_PATH=${amirstan_plugin_root}/build/lib  ##最好写进 ~/.bashrc
```

### install openbayes_torch2trt

```shell
cd openbayes_torch2trt
python setup.py develop
```



## Usage

### Convert

```python
import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
```

### Execute

We can execute the returned ``TRTModule`` just like the original PyTorch model

```python
y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))
```

### Save and load

We can save the model as a ``state_dict``.

```python
torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
```

We can load the saved model into a ``TRTModule``

```python
from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
```



## Models

We tested the converter against these models using the [test.sh](openbayes_torch2trt/test.sh) script.  You can generate the results by calling

```bash
./test.sh TEST_OUTPUT.md
```

> The results below show the throughput in FPS.  You can find the raw output, which includes latency, in the [benchmarks folder](openbayes_torch2trt/benchmarks).

| Model         | Nano (PyTorch) | Nano (TensorRT) | Xavier (PyTorch) | Xavier (TensorRT) |
| ------------- | :------------: | :-------------: | :--------------: | :---------------: |
| alexnet       |      46.4      |      69.9       |       250        |        580        |
| squeezenet1_0 |       44       |       137       |       130        |        890        |
| squeezenet1_1 |      76.6      |       248       |       132        |       1390        |
| resnet18      |      29.4      |      90.2       |       140        |        712        |
| resnet34      |      15.5      |      50.7       |       79.2       |        393        |
| resnet50      |      12.4      |      34.2       |       55.5       |        312        |
| resnet101     |      7.18      |      19.9       |       28.5       |        170        |
| resnet152     |      4.96      |      14.1       |       18.9       |        121        |
| densenet121   |      11.5      |      41.9       |       23.0       |        168        |
| densenet169   |      8.25      |      33.2       |       16.3       |        118        |
| densenet201   |      6.84      |      25.4       |       13.3       |       90.9        |
| densenet161   |      4.71      |      15.6       |       17.2       |       82.4        |
| vgg11         |      8.9       |      18.3       |       85.2       |        201        |
| vgg13         |      6.53      |      14.7       |       71.9       |        166        |
| vgg16         |      5.09      |      11.9       |       61.7       |        139        |
| vgg19         |                |                 |       54.1       |        121        |
| vgg11_bn      |      8.74      |      18.4       |       81.8       |        201        |
| vgg13_bn      |      6.31      |      14.8       |       68.0       |        166        |
| vgg16_bn      |      4.96      |      12.0       |       58.5       |        140        |
| vgg19_bn      |                |                 |       51.4       |        121        |



## How to add (or override) a converter

Here we show how to add a converter for the ``ReLU`` module using the TensorRT
python API.

```python
import tensorrt as trt
from torch2trt import tensorrt_converter

@tensorrt_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_activation(input=input._trt, type=trt.ActivationType.RELU)  
    output._trt = layer.get_output(0)
```

The converter takes one argument, a ``ConversionContext``, which will contain
the following

* ``ctx.network`` - The TensorRT network that is being constructed.

* ``ctx.method_args`` - Positional arguments that were passed to the specified PyTorch function.  The ``_trt`` attribute is set for relevant input tensors.
* ``ctx.method_kwargs`` - Keyword arguments that were passed to the specified PyTorch function.
* ``ctx.method_return`` - The value returned by the specified PyTorch function.  The converter must set the ``_trt`` attribute where relevant.

Please see [this folder](openbayes_torch2trt/torch2trt/converters) for more examples.



## How to add (or override) a plugin

1.在训练模型时自定义网络层

```python
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x
```

2.在openbayes_plugin中编写插件，具体可见此[示例](openbayes_plugin/src/plugin/CenteredLayerPlugin/)

3.python调用写好的插件

+ 在openbayes_torch2trt/torch2trt/plugins/中创建插件，[示例](openbayes_torch2trt/torch2trt/converters/CenteredLayer.py)
+ 在openbayes_torch2trt/torch2trt/converters/中添加此插件的converter，[示例](openbayes_torch2trt/torch2trt/plugins/create_centeredlayer_plugin.py)

4.可以[在此](openbayes_torch2trt/demo.py)参考最终的调用流程，