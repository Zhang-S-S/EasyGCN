import EasyGCN.classes
import EasyGCN.nn

from EasyGCN.classes import *
from EasyGCN.nn import *

from EasyGCN.nn import GCNConv
from EasyGCN.nn import GATConv
from EasyGCN.nn import GraphSAGEConv

__all__ = ['GCNConv','GATConv', 'GraphSAGEConv']

def __getattr__(name):
    print(f"attr {name} doesn't exist!")
