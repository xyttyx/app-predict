import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans

import numpy as np

from Models import ModelLSTM, ModelAttention
from config import config_attn as config

app_number = config.app_number
user_number = config.user_number
app_embedding_dim = config.app_embedding_dim
user_embedding_dim = config.user_embedding_dim
seq_length = config.seq_length
use_poi = True
poi_embedding = None
'''
model = ModelAttention(
            app_number=app_number,
            user_number=user_number,
            app_embedding_dim=app_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            seq_length=seq_length,
            use_poi=use_poi,
            poi_embedding=poi_embedding,
        ).to(device='cpu')

model.load_state_dict(torch.load("./Dataset/model_attn_with_poi_newest.pth", map_location=torch.device('cpu')))'''

with open("Dataset/App2Category.txt", "r") as f:
    catagory = f.readlines()
    catagory = [line.strip().split() for line in catagory]
    catagory = [int(line[1]) for line in catagory]
    catagory = np.array(catagory)

app_embedding = torch.load('./Dataset/app_embeddings.pt', weights_only=False).cpu().numpy()
model_state = torch.load('./Save/model/model_lstm_with_poi_newest.pth', weights_only=False,map_location="cpu")
weights = model_state["weights"]
phases = model_state["phases"]
time_embedding = weights * torch.tensor(list(range(240))).unsqueeze(-1) + phases
time_embedding[:,1:] = torch.sin(time_embedding[:,1:])
time_embedding = time_embedding.numpy()
time_embedding = torch.randn((240,54),)  * 10
time_catagory = []
for i in range(1,9):
    tmp = torch.ones((30,)) * i
    time_catagory.append(tmp)
time_catagory = torch.cat(time_catagory).numpy()
X_scaled = time_embedding
labels = time_catagory

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
labels = time_catagory
print("Using ground truth labels:")
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score = {score:.3f}")
score = calinski_harabasz_score(X_scaled, labels)
print(f"CH Index = {score:.1f}")
