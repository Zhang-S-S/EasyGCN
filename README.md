# EasyGCN: A Memory-Aware High-Performance Library for Graph Convolutional Networks on CPUs


This repository contains the source code for **"EasyGCN: A Memory-Aware High-Performance Library for Graph Convolutional Networks on CPUs"**. 

---

## 1. Getting Started Instructions.

### Step 1.1: Clone the Repository
```python
git clone [https://anonymous.4open.science/r/EasyGCN-0DFE](https://anonymous.4open.science/r/EasyGCN-0DFE)
cd EasyGCN-0DFE
```

### Step 1.2: Create a Virtual Environment
```python
conda create -n easygcn python=3.10 -y
conda activate easygcn
```

### Step 1.3: Install Core Dependencies
```python
# Install EasyGCN and other required Python packages
pip install EasyGCN
```

### Step 1.4: Install Baseline Libraries
```python
# Install PyG 2.7.0
pip install torch_geometric

# Install DGL 2.4.0
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# Install CogDL 0.6
pip install cogdl
```
