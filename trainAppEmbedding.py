import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Models import DeepWalkWeighted
from utils import AppDataset

import pickle

if os.path.exists("./Dataset/word2vec_data.pt"):
    word2vec_data = torch.load("./Dataset/word2vec_data.pt")
else:
    data = AppDataset(length = 7)
    word2vec_data = []
    for i in range(len(data)):
        tmp = data[i][0]
        apps = tmp[:,4]
        app = apps[3]
        for j in range(len(apps)):
            app2 = apps[j]
            if app2 != app:
                word2vec_data.append(torch.tensor([app, app2],dtype=torch.int16))
    word2vec_data = torch.stack(word2vec_data)
    torch.save(word2vec_data, "./Dataset/word2vec_data.pt")

HeteroGraph = pickle.load(open("./Dataset/graph_data.pkl", "rb"))
edge_index_app = HeteroGraph['app', 'to', 'app'].edge_index
edge_weight_app = HeteroGraph['app', 'to', 'app'].attr
app_num = 1696
first_time = False
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if first_time:
    with open("Dataset/App2Category.txt", "r") as f:
        catagory = f.readlines()
        catagory = [line.strip().split() for line in catagory]
        catagory = [int(line[1]) for line in catagory]
        catagory = torch.tensor(catagory)
    catagory_embedding = torch.rand((20,200)) * 2 - 1
    pretrained_weight = torch.zeros((1696,200))
    for i in range(0,1696):
        pretrained_weight[i] = (catagory_embedding[catagory[i]] + torch.rand((200,)))
else:
    pretrained_weight = torch.load("./Dataset/app_embeddings.pt", weights_only=False)


model = DeepWalkWeighted(
    num_nodes=app_num,
    edge_index=edge_index_app,
    edge_weight=edge_weight_app,
    embedding_size=200,
    pretrained_weight = pretrained_weight,
    walk_length=5,
    num_walks=50,
    batch_size=256,
    lr = 1e-4,
    epochs=20,
    sampling_times=5,
    device=device
)

embeddings = model.train()
torch.save(embeddings, "./Dataset/app_embeddings.pt")
