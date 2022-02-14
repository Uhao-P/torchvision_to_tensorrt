import ctypes
import os
import os.path as osp

dir_path = osp.join(os.path.expanduser('~'), 'space/trt_plugin/build/lib/')

if not osp.exists(dir_path):
    if 'OPENBAYES_LIBRARY_PATH' in os.environ:
        dir_path = os.environ['OPENBAYES_LIBRARY_PATH']
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))


def load_plugin_library():
    ctypes.CDLL(osp.join('/opt/openbayes_plugin/build/lib', 'libopenbayes_plugin.so'))
