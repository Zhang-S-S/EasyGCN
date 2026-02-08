import easygraph as eg
import torch
import numpy as np
import ctypes
import metis
from torch_sparse import SparseTensor
from typing import List, Tuple, Dict

class Graph(eg.Graph):
    @property
    def edge_index(self):
        import torch
        if "edge_index" not in self.cache:
            edge_list = [(u, v) for u, neighbors in self._adj.items() for v in neighbors]
            self_loops = [(u, u) for u in self._adj.keys()]
            edge_list += self_loops
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
            self.cache["edge_index"] = edge_index
        return self.cache["edge_index"]

    @property
    def norm_info(self):
        import torch

        if "norm_info" not in self.cache:
            edge_index = self.edge_index
            row, col = edge_index
            deg_dict = self.degree()
            deg = torch.tensor([deg_dict[i] for i in range(len(self.nodes))], dtype=torch.float32)
            # deg = self.degree_tensor(col, len(self.nodes), dtype=torch.float32)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            self.cache["norm_info"] = (row, col, norm)
        return self.cache["norm_info"]
    
    @property
    def adj_t(self):

        # if "adj_t" not in self.cache:
        row, col, norm = self.norm_info
        N = len(self.nodes)

        # self.cache["adj_t"] = SparseTensor(row=row, col=col, value=norm, sparse_sizes=(len(self.nodes), len(self.nodes)))

        return SparseTensor(row=row, col=col, value=norm, sparse_sizes=(len(self.nodes), len(self.nodes)))

    def build_adj_gp(self, nparts: int = 4, prune_block_size: int = 64, prune_threshold: int = 8):
        if "adj_gp" not in self.cache or "adj_gp_t" not in self.cache:
            row, col, norm = self.norm_info
            N = len(self.nodes)
            nnz = len(row)
 
            adj_list = [[] for _ in range(N)] # [优化] 存在优化的可能
            for u, v in zip(row, col):
                adj_list[u].append(v)

            # --- 修改后的划分预处理 ---
            idx_t = ctypes.c_int32
            xadj = (idx_t*(N+1))()      # shape: (N+1,)
            adjncy =  (idx_t*(nnz))()  # shape: (nnz,)
            adjwgt = (idx_t*nnz)()  # shape: (nnz,)
            xadj[0] = ptr = 0
            for i, adj in enumerate(adj_list):
                for j in adj:
                    adjncy[ptr] = j
                    adjwgt[ptr] = 1
                    ptr += 1
                xadj[i+1] = ptr
            
            _, parts = metis.part_graph({
                'nvtxs': idx_t(N),  # 节点数
                'ncon': idx_t(1),
                'xadj': xadj,
                'adjncy': adjncy,
                'vwgt': None, # 节点权重
                'vsize': None,  # 节点大小 默认 None
                'adjwgt': adjwgt # 边权重
            }, nparts=nparts)

            # 替代 concatenate 操作
            part_to_nodes = [[] for _ in range(nparts)]
            for idx, p in enumerate(parts):
                part_to_nodes[p].append(idx)
            perm = np.concatenate(part_to_nodes)

            inv_perm = np.argsort(perm)

            row2 = inv_perm[row]
            col2 = inv_perm[col]
            adj_gp = SparseTensor(
                row=torch.tensor(row2, dtype=torch.long),
                col=torch.tensor(col2, dtype=torch.long),
                value=norm,
                sparse_sizes=(N, N)
            )
            
            adj_gp_t = SparseTensor(
                row=torch.tensor(col2, dtype=torch.long), # 交换：原来的 col2 变成 row
                col=torch.tensor(row2, dtype=torch.long), # 交换：原来的 row2 变成 col
                value=norm, # 对称归一化图中值不变；若为有向图需注意
                sparse_sizes=(N, N),
                is_sorted=False # 显式告诉它未排序，让它内部处理 CSR 排序
            )

            
            self.cache['adj_gp'] = adj_gp
            self.cache['adj_gp_t'] = adj_gp_t
            self.cache['gp_perm'] = torch.tensor(perm)
            self.cache['gp_inv_perm'] = torch.tensor(inv_perm)
            
        return
