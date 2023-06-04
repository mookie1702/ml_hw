import imp
import os.path as osp


def load(name):
    """
    函数作用: 加载指定名称的模块.
    """
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source("", pathname)
