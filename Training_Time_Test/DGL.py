import time
import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import psutil
import statistics
from sklearn.metrics import f1_score
from tqdm import tqdm 
from txtReader import TxtGraphReader

from torch_geometric.datasets import Coauthor, Planetoid, Reddit
import torch_geometric.transforms as T
import dgl
import dgl.nn.pytorch as dglnn
from ogb.nodeproppred import PygNodePropPredDataset

# -------------------- Setup --------------------
BACKEND = 'DGL'
DEVICE = torch.device('cpu')
SEED = 42
EPOCHS = 200
RR = 30

DATASETS = [
    ('Coauthor', 'CS'),
    ('Coauthor', 'Physics'),
    ('Planetoid', 'Cora'),
    ('Planetoid', 'Citeseer'),
    ('Planetoid', 'PubMed'),
    ('ogb', 'ogbn-arxiv'),
    ('txt', 'Web_BerkStan'),
    ('txt', 'soc-pokec'),
    ('reddit', 'Reddit'),
    ('ogb', 'ogbn-products'),
]


# -------------------- random_seed --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)

set_seed(SEED)

_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

transform = T.NormalizeFeatures()


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, out_dim)

    def forward(self, graph, features):
        x = F.relu(self.conv1(graph, features))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(graph, x)
        return x

Result_Chart = {}

def Show_Result(dataset_name, RR, All_total_train_time, All_epoch_times, All_forward_times, All_backward_times, peak_memory_mb, All_test_acc, ALL_f1):
    print("\n======= Result =======")
    print(f" Framework: {BACKEND}")
    print(f" Dataset: {dataset_name}")
    print(f" total_time: {sum(All_total_train_time)/RR:.3f} 秒 ({[round(_, 3) for _ in All_total_train_time]})")
    print(f" STD_time: {statistics.stdev(All_total_train_time):.3f} 秒")
    print(f" Avg_epoch_time: {sum(All_epoch_times)/RR*1000:.3f} ms")
    print(f" Avg_forward_time: {sum(All_forward_times)/RR*1000:.3f} ms")
    print(f" Avg_backward_time: {sum(All_backward_times)/RR*1000:.3f} ms")
    print(f" Accuracy: {sum(All_test_acc)/RR:.4f}")
    print(f" STD_acc: {statistics.stdev(All_test_acc):.4f}")
    print(f" F1-score: {sum(ALL_f1)/RR:.4f}")
    Result_Chart[dataset_name] = {
        "Total_Train_Time": sum(All_total_train_time)/RR,
        "Std_Train_Time": statistics.stdev(All_total_train_time),
        "Accuracy": sum(All_test_acc)/RR,
        "Std_Accuracy": statistics.stdev(All_test_acc),
    }
    
def main():
    root = "./dataset/"
    for backend_type, dataset_name in DATASETS:
        print(f"\n================= Dataset: {dataset_name} =================")

        if backend_type == 'Coauthor':
            dataset = Coauthor(root=root, name=dataset_name)
        elif backend_type == 'Planetoid':
            dataset = Planetoid(root=root, name=dataset_name, transform=transform)
        elif backend_type == 'ogb':
            dataset = PygNodePropPredDataset(name=dataset_name, root=root)
        elif backend_type == 'reddit':
            dataset = Reddit(root=root + 'Reddit')
        elif backend_type == 'txt':
            dataset = TxtGraphReader(root=root, name=dataset_name)
        else:
            raise ValueError(f"Unknown dataset type {backend_type}")

        data = dataset[0]
        num_nodes = data.num_nodes

        data.x = data.x.float()
        data.y = data.y.long()

        if backend_type == 'ogb':  
            split_idx = dataset.get_idx_split()
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[split_idx['train']] = True
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[split_idx['valid']] = True
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[split_idx['test']] = True
        elif backend_type in ['Planetoid', 'reddit']:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask

        elif backend_type == 'Coauthor': 
            torch.manual_seed(42) 
            indices = torch.randperm(num_nodes)

            train_end = int(0.6 * num_nodes)
            val_end = int(0.8 * num_nodes)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[indices[:train_end]] = True
            val_mask[indices[train_end:val_end]] = True
            test_mask[indices[val_end:]] = True

        elif backend_type == 'txt':
       
            torch.manual_seed(42) 
            indices = torch.randperm(num_nodes)

            train_end = int(0.6 * num_nodes)
            val_end = int(0.8 * num_nodes)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[indices[:train_end]] = True
            val_mask[indices[train_end:val_end]] = True
            test_mask[indices[val_end:]] = True

        else:
            raise ValueError(f"Unknown dataset type {backend_type}")

        if dataset_name == 'ogbn-arxiv':
            HIDDEN_DIM = 256
        else:
            HIDDEN_DIM = 16

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(f"Train nodes: {train_mask.sum().item()} | Val nodes: {val_mask.sum().item()} | Test nodes: {test_mask.sum().item()}")

        src, dst = data.edge_index
        g = dgl.graph((src, dst), num_nodes=data.num_nodes)
        g = dgl.add_self_loop(g)
        g = g.to(DEVICE)
        g.ndata['feat'] = data.x.to(DEVICE)
        
        try:
            data = data.to(DEVICE)
        except:
            pass

        try:
            num_node_features = dataset.num_node_features
        except:
            num_node_features = data.x.shape[1]
        try:
            num_classes = dataset.num_classes
        except:
            num_classes = int(data.y.max().item()) + 1

        All_forward_times, All_backward_times, All_epoch_times = [], [], []
        All_total_train_time, All_test_acc, ALL_f1 = [], [], []

        x_orig = data.x.clone()
        y_orig = data.y.clone()
        train_mask_orig = train_mask.clone()
        val_mask_orig = val_mask.clone()
        test_mask_orig = test_mask.clone()
        for R in tqdm(range(RR + 1)):
            
            model = GCN(num_node_features, HIDDEN_DIM, num_classes).to(DEVICE)
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            # model = torch.compile(model, backend="inductor")
            optimizer = torch.optim.Adam([
                {'params': model.conv1.parameters(), 'weight_decay': 5e-4}, 
                {'params': model.conv2.parameters(), 'weight_decay': 0.0}  
            ], lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            early_stop_counter = 0

            LOSS_LIST, LOSS_LIST_VALID, LOSS_LIST_TEST = [], [], []
            forward_times, backward_times, epoch_times = [], [], []

            process = psutil.Process(os.getpid())
            peak_memory_mb = process.memory_info().rss / 1024 / 1024

            train_start = time.perf_counter()

            # for epoch in tqdm(range(1, EPOCHS+1)):
            for epoch in range(1, EPOCHS+1):
                model.train()
                optimizer.zero_grad()
                start_fwd = time.perf_counter()
                out = model(g, data.x)
                end_fwd = time.perf_counter()

                loss = criterion(out[data.train_mask], data.y[data.train_mask].squeeze())
                loss_val = criterion(out[data.val_mask], data.y[data.val_mask].squeeze())
                loss_test = criterion(out[data.test_mask], data.y[data.test_mask].squeeze())

                start_bwd = time.perf_counter()
                loss.backward()
                optimizer.step()
                end_bwd = time.perf_counter()

                forward_times.append(end_fwd - start_fwd)
                backward_times.append(end_bwd - start_bwd)
                epoch_times.append(end_fwd - start_fwd + end_bwd - start_bwd)
                LOSS_LIST.append(loss.item())
                LOSS_LIST_VALID.append(loss_val.item())
                LOSS_LIST_TEST.append(loss_test.item())

                # early stopping
                # if loss_val.item() < best_val_loss:
                #     best_val_loss = loss_val.item()
                #     early_stop_counter = 0
                # else:
                #     early_stop_counter += 1

                # if early_stop_counter >= EARLY_STOP_WINDOW:
                #     break

                current_memory_mb = process.memory_info().rss / 1024 / 1024
                peak_memory_mb = max(peak_memory_mb, current_memory_mb)

            train_end = time.perf_counter()
            total_train_time = train_end - train_start

            model.eval()
            with torch.no_grad():
                out = model(g, data.x)
                pred = out.argmax(dim=1)
                test_acc = (pred[data.test_mask] == data.y[data.test_mask].squeeze()).sum().item() / data.test_mask.sum().item()
                macro_f1 = f1_score(data.y[data.test_mask].squeeze(), pred[data.test_mask], average='macro')

            All_forward_times.append(sum(forward_times)/len(forward_times))
            All_backward_times.append(sum(backward_times)/len(backward_times))
            All_epoch_times.append(sum(epoch_times)/len(epoch_times))
            All_total_train_time.append(total_train_time)
            All_test_acc.append(test_acc)
            ALL_f1.append(macro_f1)


        Show_Result(dataset_name, RR, All_total_train_time[-RR:], All_epoch_times[-RR:], All_forward_times[-RR:], All_backward_times[-RR:], peak_memory_mb, All_test_acc[-RR:], ALL_f1[-RR:])
        
if __name__ == "__main__":
    main()