import torch
import torch.nn as nn
# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(GCNConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):  

#         out = g.adj_t.matmul(x @ self.weight)

#         if self.bias is not None:
#             out += self.bias

#         return out

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


## 临时使用 edge_index训练
# import torch
# import torch.nn as nn

# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(GCNConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()
#         self.cached_adj = None
        
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):  
        
#         # [优化]：检查是否已经缓存过，如果缓存过直接用，不再 to_dense
#         if self.cached_adj is None:
#             # 只有第一次运行时执行，转换并存入 self
#             # print("正在将稀疏矩阵转换为稠密矩阵 (仅一次)...")
#             self.cached_adj = g.L_GCN.to_dense()

#         x_trans = x @ self.weight
        
#         # 使用缓存好的矩阵
#         out = self.cached_adj @ x_trans 

#         if self.bias is not None:
#             out += self.bias

#         return out
    

## The version of the GP 

# import torch
# import torch.nn as nn

# class GCNConv(nn.Module):
#     '''
#     GCN with graph partition version
#     '''
#     def __init__(self, in_channels, out_channels, bias=True):
#         super(GCNConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):  

#         out = g.cache['adj_gp'].matmul(x @ self.weight)

#         if self.bias is not None:
#             out += self.bias
#         return out

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)



## The version of the GP+backward 

# class FastGCNConvFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight, adj, bias=None):
#         AX = adj.matmul(x)
#         out = AX @ weight
#         # out = adj.matmul(x @ weight)
#         if bias is not None:
#             out += bias
#         # AX, x, weight, bias 是 Tensor，可以用 save_for_backward
#         ctx.save_for_backward(AX, weight, bias)
#         # adj 是稀疏矩阵，直接赋值给 ctx
#         ctx.adj = adj
#         return out

#     @staticmethod
#     def backward(ctx, grad_out):
#         AX, weight, bias = ctx.saved_tensors
#         adj = ctx.adj

#         grad_x = grad_w = grad_b = None
#         grad_w = AX.T @ grad_out
#         grad_x = adj.matmul(grad_out @ weight.T)
#         if bias is not None:
#             grad_b = grad_out.sum(0)
#         return grad_x, grad_w, None, grad_b
    
# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):
#         return FastGCNConvFn.apply(x, self.weight, g.cache['adj_gp'], self.bias)



### The version of C++ BK
# try:
#     import cpp_easygraph
#     HAS_CPP_BACKEND = True
# except ImportError:
#     print("Warning: cpp_easygraph module not found. Using slow Python fallback.")
#     HAS_CPP_BACKEND = False
    
# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):
#         return cpp_easygraph.upscale_gcn_forward(x, self.weight, g.cache['adj_torch'], self.bias)


###  The version of GP+BW upup
# class FastGCNConvFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight, adj, bias=None):
#         AX = adj.matmul(x) 
#         out = AX @ weight
        
#         if bias is not None:
#             out += bias
            
#         ctx.save_for_backward(AX, weight, bias)
#         ctx.adj = adj
        
#         return out

#     @staticmethod
#     def backward(ctx, grad_out):
#         AX, weight, bias = ctx.saved_tensors
#         adj = ctx.adj

#         grad_x = grad_w = grad_b = None
        
#         if ctx.needs_input_grad[1]:
#             grad_w = AX.t() @ grad_out

#         if ctx.needs_input_grad[0]:
#             grad_temp = grad_out @ weight.t()
#             grad_x = adj.t().matmul(grad_temp)

#         # 3. 计算 Bias 梯度
#         if bias is not None and ctx.needs_input_grad[3]:
#             grad_b = grad_out.sum(0)

#         return grad_x, grad_w, None, grad_b

# class GCNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)

#     def forward(self, x, g):
#         # 1. 检查 adj_gp 是否存在
#         if not hasattr(g, 'cache') or 'adj_gp' not in g.cache:
#             raise RuntimeError("EasyGraph Error: 'adj_gp' not found in graph cache. Please run g.build_adj_gp() first.")


#         return FastGCNConvFn.apply(x, self.weight, g.cache['adj_gp'], self.bias)


###  The version of GP+BW with Path selection
import torch
import torch.nn as nn

class HybridGCNConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, adj, adj_t, bias=None):
        ctx.input_dim = x.shape[1]
        ctx.output_dim = weight.shape[1]
        
        # 将 adj 直接保存在 ctx 上，而不是 save_for_backward
        # 因为 adj可能是自定义的 SparseTensor 对象，无法被 autograd 序列化
        ctx.adj = adj 
        ctx.adj_t = adj_t
        # 策略 A: 先降维 (Reddit 等)
        if ctx.input_dim > ctx.output_dim:
            XW = x.matmul(weight)
            out = adj.matmul(XW)
            ctx.save_for_backward(x, weight, XW) 
            # ctx.save_for_backward(x, weight, None)
            ctx.path = "A"
            
        # 策略 B: 先传播 (Cora / OGBN-Products 等)
        else:
            AX = adj.matmul(x)
            out = AX.matmul(weight)
            ctx.save_for_backward(x, weight, AX)
            ctx.path = "B"

        if bias is not None:
            out += bias
            ctx.has_bias = True
        else:
            ctx.has_bias = False

        return out

    @staticmethod
    def backward(ctx, grad_out):
        # 修改点 3: 从 ctx.adj 获取 adj，从 saved_tensors 获取其他张量
        x, weight, saved_tensor = ctx.saved_tensors
        adj = ctx.adj
        adj_t = ctx.adj_t
        
        grad_x = grad_w = grad_b = None
        
        # --- 路径 A 反向 ---
        if ctx.path == "A":
            XW = saved_tensor
            
            # 1. grad_XW (Sparse MM)
            # grad_XW = adj.t().matmul(grad_out)
            grad_XW = adj_t.matmul(grad_out)
            # 2. grad_w (Dense MM)
            if ctx.needs_input_grad[1]:
                grad_w = x.t().matmul(grad_XW)
            
            # 3. grad_x (Dense MM)
            if ctx.needs_input_grad[0]:
                grad_x = grad_XW.matmul(weight.t())

        # --- 路径 B 反向 ---
        else:
            AX = saved_tensor
            
            # 1. grad_w (Dense MM)
            if ctx.needs_input_grad[1]:
                grad_w = AX.t().matmul(grad_out)
                
            if ctx.needs_input_grad[0]:
                grad_temp = grad_out.matmul(weight.t())
                # 2. grad_x (Sparse MM)
                # grad_x = adj.t().matmul(grad_temp)
                grad_x = adj_t.matmul(grad_temp)

        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_b = grad_out.sum(0)

        # 注意：adj (第2个参数) 通常不需要梯度，返回 None
        return grad_x, grad_w, None, None, grad_b

class GCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, g):
        # 假设 g.cache['adj_gp'] 是处理好的稀疏矩阵
        return HybridGCNConvFn.apply(x, self.weight, g.cache['adj_gp'], g.cache['adj_gp_t'], self.bias)