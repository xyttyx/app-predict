import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np

from Models import ModelLSTM
from config import config_lstm

app_number = config_lstm.app_number
user_number = config_lstm.user_number
app_embedding_dim = config_lstm.app_embedding_dim
user_embedding_dim = config_lstm.user_embedding_dim
seq_length = config_lstm.seq_length
use_poi = True
poi_embedding = None

model = ModelLSTM(
            app_number=app_number,
            user_number=user_number,
            app_embedding_dim=app_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            seq_length=seq_length,
            use_poi=use_poi,
            poi_embedding=poi_embedding,
        )

model.load_state_dict(torch.load("./Save/model/model_lstm_with_poi_newest.pth"))
app_embedding = model.app_embedding.weight.data.cpu().numpy()

with open("Dataset/App2Category.txt", "r") as f:
    catagory = f.readlines()
    catagory = [line.strip().split() for line in catagory]
    catagory = [int(line[1]) for line in catagory]
    catagory = np.array(catagory)

X_scaled = app_embedding

print("Silhouette Score and Calinski-Harabasz Index for different number of clusters:")
print("----------------------------------------------------------")
for n_clusters in range(15, 21):
    print(f"n_clusters={n_clusters}:")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score = {score:.3f}")
    score = calinski_harabasz_score(X_scaled, labels)
    print(f"CH Index = {score:.1f}")

print("----------------------------------------------------------")
labels = catagory
print("Using ground truth labels:")
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score = {score:.3f}")
score = calinski_harabasz_score(X_scaled, labels)
print(f"CH Index = {score:.1f}")

# 轮廓系数-0.437，表明真实标签的聚类效果不好，可能是因为嵌入的维度过高，或者数据本身的分布不适合聚类。