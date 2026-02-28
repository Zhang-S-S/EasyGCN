import time
import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import psutil
from sklearn.metrics import f1_score
from tqdm import tqdm
import statistics
from txtReader import TxtGraphReader
from torch_geometric.datasets import Coauthor, Planetoid, Reddit
import torch_geometric.transforms as T
from cogdl.models.nn import GCN 
import torch.nn.functional as F
from cogdl.data import Graph
from ogb.nodeproppred import PygNodePropPredDataset
# -------------------- Setup --------------------
BACKEND = 'Cogdl'
DEVICE = torch.device('cpu') 
SEED = 42
EPOCHS = 200
DROPOUT = 0.5
EARLY_STOP_WINDOW = 10
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(SEED)

_load = torch.load
def load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load(*args, **kwargs)
torch.load = load

transform = T.NormalizeFeatures()

Result_Chart = {}

def Show_Result(dataset_name, RR, All_total_train_time, All_epoch_times, All_forward_times, All_backward_times, peak_memory_mb, All_test_acc, ALL_f1):
    print("\n======= Result =======")
    print(f"🔹 Framework: {BACKEND}")
    print(f"🔹 Dataset: {dataset_name}")
    print(f"🔥 total_time: {sum(All_total_train_time)/RR:.3f} 秒 ({[round(_, 3) for _ in All_total_train_time]})")
    print(f"🔥 STD_time: {statistics.stdev(All_total_train_time):.3f} 秒")
    print(f"⏩ Avg_epoch_time: {sum(All_epoch_times)/RR*1000:.3f} ms")
    print(f"🔁 Avg_forward_time: {sum(All_forward_times)/RR*1000:.3f} ms")
    print(f"↩️ Avg_backward_time: {sum(All_backward_times)/RR*1000:.3f} ms")
    print(f"🎯 Accuracy: {sum(All_test_acc)/RR:.4f}")
    print(f"🎯 STD_acc: {statistics.stdev(All_test_acc):.4f}")
    print(f"🎯 F1-score: {sum(ALL_f1)/RR:.4f}")
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

        if not hasattr(data, 'edge_index'):
            raise ValueError(f"Data object missing 'edge_index' (dataset: {dataset_name})")
        
        edge_index = data.edge_index

        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index).long().to(DEVICE)
        else:
            edge_index = edge_index.long().to(DEVICE)
        
        row = edge_index[0]  # shape: [E]
        col = edge_index[1]  # shape: [E]


        g = Graph(
            x=data.x,
            row=row,          
            col=col,          
            y=data.y,
            num_nodes=num_nodes
        )  

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
        # data = data.to(DEVICE)
        try:
            num_node_features = dataset.num_node_features
        except:
            num_node_features = data.x.shape[1]
        try:
            num_classes = dataset.num_classes
        except:
            num_classes = int(data.y.max().item()) + 1

        All_forward_times = []
        All_backward_times = []
        All_epoch_times = []
        All_total_train_time = []
        All_test_acc = []
        ALL_f1 = []


        for R in tqdm(range(RR+1)):
            model = GCN(
                in_feats=num_node_features,
                hidden_size=HIDDEN_DIM,
                out_feats=num_classes,
                num_layers=2,
                dropout=DROPOUT,
                activation="relu",
                residual=False,
                norm=None
            ).to(DEVICE)   

            optimizer = torch.optim.Adam([
                {'params': model.layers[0].parameters(), 'weight_decay': 5e-4},  
                {'params': model.layers[1].parameters(), 'weight_decay': 0.0} 
            ], lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
            peak_memory_mb = memory_usage_mb

            forward_times = []
            backward_times = []
            epoch_times = []

            LOSS_LIST = []
            LOSS_LIST_TEST = []
            LOSS_LIST_VALID = []

            best_val_loss = float('inf')
            early_stop_counter = 0

            train_start = time.time()
            for epoch in tqdm(range(1, EPOCHS + 1)):
                model.train()
                optimizer.zero_grad()
                epoch_start = time.time()

                start_fwd = time.time() 
                out = model(g) 
                end_fwd = time.time()

                loss = criterion(out[data.train_mask], data.y[data.train_mask].squeeze())
                loss_test = criterion(out[data.test_mask], data.y[data.test_mask].squeeze())
                loss_valid = criterion(out[data.val_mask], data.y[data.val_mask].squeeze())

                start_bwd = time.time()
                loss.backward()
                optimizer.step()
                end_bwd = time.time()

                epoch_end = time.time()

                forward_times.append(end_fwd - start_fwd)
                backward_times.append(end_bwd - start_bwd)
                epoch_times.append(epoch_end - epoch_start)

                LOSS_LIST.append(round(loss.item(),3))
                LOSS_LIST_TEST.append(round(loss_test.item(),3))
                LOSS_LIST_VALID.append(round(loss_valid.item(),3))

                # # early stopping
                # if loss_valid.item() < best_val_loss:
                #     best_val_loss = loss_valid.item()
                #     early_stop_counter = 0
                # else:
                #     early_stop_counter += 1

                # if early_stop_counter >= EARLY_STOP_WINDOW:
                #     # print(f"Early stopping at epoch {epoch}. Validation loss has not decreased for {EARLY_STOP_WINDOW} epochs.")
                #     break

                current_memory_mb = process.memory_info().rss / 1024 / 1024
                peak_memory_mb = max(peak_memory_mb, current_memory_mb)

            train_end = time.time()
            total_train_time = train_end - train_start

            @torch.no_grad()
            def evaluate(model, data, mask_key='test_mask'):
                model.eval()
                out = model(g)
                mask = getattr(data, mask_key)
                pred = out.argmax(dim=1)
                correct = (pred[mask] == data.y[mask].squeeze()).sum()
                acc = int(correct) / int(mask.sum())
                macro_f1 = f1_score(data.y[mask].squeeze(), pred[mask], average='macro')
                return acc, macro_f1

            test_acc, f1 = evaluate(model, data)

            avg_fwd = sum(forward_times) / len(forward_times)
            avg_bwd = sum(backward_times) / len(backward_times)
            avg_epoch = sum(epoch_times) / len(epoch_times)

            All_forward_times.append(avg_fwd)
            All_backward_times.append(avg_bwd)
            All_epoch_times.append(avg_epoch)
            All_total_train_time.append(total_train_time)
            All_test_acc.append(test_acc)
            ALL_f1.append(f1)
        
        Show_Result(dataset_name, RR, All_total_train_time[-RR:], All_epoch_times[-RR:], All_forward_times[-RR:], All_backward_times[-RR:], peak_memory_mb, All_test_acc[-RR:], ALL_f1[-RR:])

if __name__ == "__main__":
    main()