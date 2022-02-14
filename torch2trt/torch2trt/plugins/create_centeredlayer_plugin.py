import numpy as np
import tensorrt as trt


def create_centeredlayer_plugin(layer_name):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'CenteredLayer', '1', '')

    pfc = trt.PluginFieldCollection()

    return creator.create_plugin(layer_name, pfc)
