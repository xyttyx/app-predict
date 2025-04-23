import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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

model = ModelAttention(
            app_number=app_number,
            user_number=user_number,
            app_embedding_dim=app_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            seq_length=seq_length,
            use_poi=use_poi,
            poi_embedding=poi_embedding,
        )

model.load_state_dict(torch.load("./Save/model/model_attn_with_poi_newest.pth", map_location=torch.device('cpu')))
app_embedding = model.app_embedding.weight.data.cpu().numpy()

with open("Dataset/App2Category.txt", "r") as f:
    catagory = f.readlines()
    catagory = [line.strip().split() for line in catagory]
    catagory = [int(line[1]) for line in catagory]
    catagory = np.array(catagory)

count = np.bincount(catagory)
print("Cluster counts:")
for i, c in enumerate(count):
    print(f"Category {i}: {c} samples")
print("----------------------------------------------------------")

tsne_2d = TSNE(n_components=2, random_state=42)
embedding_2d = tsne_2d.fit_transform(app_embedding)

# 画图
plt.figure(figsize=(10, 10))
plt.title("t-SNE visualization of app embeddings")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.scatter(embedding_2d[:,0], embedding_2d[:, 1], c=catagory, cmap='rainbow', s=10)
plt.show()

tsne_3d = TSNE(n_components=3, random_state=42)
embedding_3d = tsne_3d.fit_transform(app_embedding)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("t-SNE visualization of app embeddings")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.scatter(embedding_3d[:,0], embedding_3d[:, 1], embedding_3d[:, 2], c=catagory, cmap='rainbow', s=10)
plt.show()

kmeans = KMeans(n_clusters=20, random_state=42)
labels = kmeans.fit_predict(app_embedding)

count = np.bincount(labels)
print("Cluster counts:")
for i, c in enumerate(count):
    print(f"Cluster {i}: {c} samples")
print("----------------------------------------------------------")

plt.figure(figsize=(10, 10))
plt.title("KMeans clustering of app embeddings")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.scatter(embedding_2d[:,0], embedding_2d[:, 1], c=labels, cmap='rainbow', s=10)
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("t-SNE visualization of app embeddings")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.scatter(embedding_3d[:,0], embedding_3d[:, 1], embedding_3d[:, 2], c=labels, cmap='rainbow', s=10)
plt.show()