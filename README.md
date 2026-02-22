# EasyGCN

📁 Project Structure

data/: Directory for downloading and preprocessing benchmark datasets (e.g., ogbn-arxiv, Reddit, soc-pokec).

easygcn/: The core library source code.


partition/: Graph partitioning modules using METIS.


format/: Automatic CSR-COO hybrid format converter.


nn/: Dimension-based adaptive GCN layer implementations.

scripts/: Utility scripts for data processing.

run_node_cls.py: Main execution script for node classification experiments.


run_link_pred.py: Main execution script for link prediction tasks.


run_bot_detection.py: Execution script for the Mastodon social bot detection case study.

🚀 Quick Start
1. Node Classification
To evaluate the standard 2-layer GCN model on benchmark datasets (e.g., cora, pubmed, ogbn-products), run the following command. The script will automatically apply graph partitioning and hybrid execution:

2. Link Prediction
To evaluate training efficiency on link prediction tasks utilizing a three-layer GCN , you can run it on OGB datasets (ogbl-ddi, ogbl-collab, ogbl-citation2):

3. Mastodon Social Bot Detection (Case Study)
To reproduce the practical case study on decentralized online social networks using the FediData dataset (12,548 nodes, 1,297,157 edges):

📝 License & Citation
This project is currently under double-blind peer review for KDD '26. Please do not distribute.

If you use this code in your research, please cite our paper:
