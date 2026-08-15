import EasyGCN.classes
import EasyGCN.nn

from EasyGCN.classes import *
from EasyGCN.nn import *

from EasyGCN.nn import GCNConv

__all__ = ['GCNConv']

def __getattr__(name):
    print(f"attr {name} doesn't exist!")
