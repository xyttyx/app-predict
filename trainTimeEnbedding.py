import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm

from Models import DeepWalkWeighted, SkipGram
import pickle

HeteroGraph = pickle.load(open("./Dataset/graph_data.pkl", "rb"))
edge_time2app = HeteroGraph['time-app'].edge_index
edge_app2time = HeteroGraph['app-time'].edge_index

first_time = False
app_num = 1696
time_num = 240
batch_size = 256
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if first_time:
    pretrained_weight = torch.randn((240,50))
else:
    pretrained_weight = torch.load("./Dataset/time_embeddings.pt", weights_only=False)

edge = set()
for i in tqdm(range(240)):
    time1 = i
    apps = edge_time2app[1][edge_time2app[0]==time1]
    for app in apps:
        times2 = edge_app2time[1][edge_app2time[0]==app]
        for time2 in times2:
            edge.add((time1,int(time2)))
#所有的边都存在

seq = []
for i in tqdm(range(100000)):
    time1 = random.choice(torch.range(0,239))
    apps = edge_time2app[1][edge_time2app[0]==time1]
    app = random.choice(apps)
    time2 = edge_app2time[1][edge_app2time[0]==app]
    time2 = random.choice(time2)
    seq.append(torch.tensor([time1,time2]))
seq = torch.stack(seq).to(dtype=torch.long,device=device)

dataloader = torch.utils.data.DataLoader(seq,batch_size=batch_size,shuffle=True)
model = SkipGram(240,50).to(device)
loss_fn = nn.BCELoss()

optimizer = optim.Adam(model.parameters(),lr=1e-3)

# 训练循环
for epoch in range(epochs):
    total_loss = 0
    
    for data in tqdm(dataloader):
        data = data.T
        batch_targets, batch_contexts = data[0], data[1]
        pos_scores = model(batch_targets, batch_contexts)
        pos_labels = torch.ones_like(pos_scores)
        
        neg = torch.randint(0,240,(2,batch_size//2),device=device)
        neg_targets = neg[0]
        neg_contexts = neg[1]
        neg_scores = model(neg_targets, neg_contexts)
        neg_labels = torch.zeros_like(neg_scores)
        
        # 计算损失
        x = torch.cat([pos_scores, neg_scores], dim=0)
        x = F.sigmoid(x)
        y = torch.cat([pos_labels, neg_labels], dim=0)
        loss = loss_fn(x, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.get_embeddings().cpu(), "./Dataset/time_embeddings.pt")
