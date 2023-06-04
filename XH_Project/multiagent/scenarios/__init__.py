import imp
import os.path as osp


# 函数作用: 加载指定名称的模块
def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source("", pathname)
