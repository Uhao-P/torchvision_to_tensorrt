import tensorrt as trt
from torch2trt.plugins import create_centeredlayer_plugin
from torch2trt.torch2trt import (tensorrt_converter, trt_)


@tensorrt_converter('torch.nn.functional.CenteredLayer')
def convert_CenteredLayer(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    input_trt = trt_(ctx.network, input)


    plugin = create_centeredlayer_plugin('CenteredLayer_' + str(id(input)))

    layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

    output._trt = layer.get_output(0)
