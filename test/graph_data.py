# 将app2app、app2loc、app2time、appfeatures、locfeatures、timefeatures等数据保存为图数据

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle

import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData

from utils import AppDataset

with open("./Dataset/graph_data.pkl","rb") as f:
    graph = torch.load(f, weights_only=False)

dataset = AppDataset()
data = dataset.App_usage_trace_origin
app2app = []
for i in range(len(data) - 1):
    if data[i][0] == data[i + 1][0]:
        app2app.append((int(data[i][4]), int(data[i + 1][4])))
app2app_filtered = set(app2app)
app2app_filtered = list(app2app_filtered)
app2app_weight = []
for item in app2app_filtered:
    count = app2app.count(item)
    app2app_weight.append(count)
app2loc = []
for i in range(len(data)):
    app2loc.append((int(data[i][4]), int(data[i][3])))

app2loc = set(app2loc)
app2loc = list(app2loc)
app2time = []
for i in range(len(data) - 1):
    app2time.append((int(data[i][4]), int(data[i + 1][1] * 30 + data[i + 1][2])))
app2time = set(app2time)
app2time = list(app2time)

appfeatures = torch.randn(1696, 64) 
locfeatures = torch.randn(9851, 64)  
timefeatures = torch.randn(720, 64)  

data = HeteroData()
data['app'].x = appfeatures
data['loc'].x = locfeatures
data['time'].x = timefeatures
data['app', 'to', 'app'].edge_index = torch.tensor(app2app).t()
data['app', 'to', 'app'].attr = torch.tensor(app2app_weight)
data['app', 'in', 'time'].edge_index = torch.tensor(app2time).t()
data['app', 'at', 'loc'].edge_index = torch.tensor(app2loc).t()

with open("./Dataset/graph_data.pkl", "wb") as f:
    torch.save(data,f)
