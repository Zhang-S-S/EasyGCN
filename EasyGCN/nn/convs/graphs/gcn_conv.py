import torch
import torch.nn as nn

class HybridGCNConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, adj, adj_t, bias=None):
        ctx.input_dim = x.shape[1]
        ctx.output_dim = weight.shape[1]

        ctx.adj = adj 
        ctx.adj_t = adj_t
        if ctx.input_dim > ctx.output_dim:
            XW = x.matmul(weight)
            out = adj.matmul(XW)
            ctx.save_for_backward(x, weight, XW) 
            # ctx.save_for_backward(x, weight, None)
            ctx.path = "A"

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
        x, weight, saved_tensor = ctx.saved_tensors
        adj = ctx.adj
        adj_t = ctx.adj_t
        
        grad_x = grad_w = grad_b = None
        
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
        return HybridGCNConvFn.apply(x, self.weight, g.cache['adj_gp'], g.cache['adj_gp_t'], self.bias)
