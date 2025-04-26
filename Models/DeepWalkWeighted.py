import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Tuple
from tqdm import tqdm

class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        self.embeddings.weight.data.uniform_(-1, 1)
        
    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """计算目标词和上下文词的相似度"""
        # target shape: [batch_size]
        # context shape: [batch_size]
        target_embeds = self.embeddings(target)  # [batch_size, emb_dim]
        context_embeds = self.embeddings(context)  # [batch_size, emb_dim]
        
        scores = torch.sum(target_embeds * context_embeds, dim=1)  # [batch_size]
        return scores
    
    def normalize_embeddings(self):
        weights = self.embeddings.weight.data
        l2_norms = torch.norm(weights, p=2, dim=1)
        mean_l2 = torch.mean(l2_norms) + 1e-6
        normalized_weights = weights / mean_l2
        self.embeddings.weight.data.copy_(normalized_weights)
    
    def get_embeddings(self) -> torch.Tensor:
        """获取输入嵌入矩阵"""
        return self.embeddings.weight.data

class DeepWalkWeighted:
    def __init__(self, 
                 edge_index: torch.Tensor,
                 edge_weight: torch.Tensor,
                 num_nodes: int,
                 walk_length: int = 80,
                 num_walks: int = 10,
                 embedding_size: int = 128,
                 batch_size: int = 128,
                 lr: float = 0.01,
                 epochs: int = 5,
                 sampling_times = 1,
                 device=None
                 ):
        """
        
        参数:
            edge_index: 图的边索引, shape [2, num_edges]
            num_nodes: 图中节点数量
            walk_length: 每次随机游走的长度
            num_walks: 每个节点开始的随机游走次数
            window_size: Skip-gram的窗口大小
            embedding_size: 嵌入维度
            batch_size: 训练批大小
            lr: 学习率
            epochs: 训练轮数
        """
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.sampling_times = sampling_times
        self.device=torch.device("cpu") if device is None else device
        
        # 初始化Skip-gram模型
        self.model = SkipGram(num_nodes, embedding_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()
    def get_neighbors(self, node: int) -> List[int]:
        mask = self.edge_index[0] == node
        if mask.sum() == 0:
            return None
        neighbors = self.edge_index[1][mask].cpu().numpy()
        weights = self.edge_weight[mask].cpu().numpy()
        weights = np.sqrt(weights)
        weights = weights / weights.sum()
        return np.random.choice(neighbors, p=weights)
    
    def _random_walk(self, start_node: int) -> List[int]:
        """从start_node开始随机游走"""
        walk = np.zeros((self.walk_length),dtype=np.int64)
        walk[0] = start_node
        for i in range(self.walk_length - 1):
            cur_node = walk[i]
            neighbors = self.get_neighbors(cur_node)
            if neighbors is None:
                return None
            walk[i + 1] = neighbors
        return walk
    
    def generate_walks(self):
        """为所有节点生成随机游走序列"""
        walks = []
        print("generate walks")
        for _ in tqdm(range(self.num_walks)):
            for node in range(self.num_nodes):
                walk_seq = self._random_walk(node)
                if walk_seq is not None:
                    walks.append(walk_seq)
        return np.stack(walks)
    
    def generate_training_data(self, walks) -> Tuple[torch.Tensor, torch.Tensor]:
        """从随机游走生成训练数据(目标词-上下文词对)"""
        targets = []
        contexts = []
        for i in range(self.walk_length - 1):
            for j in range(i + 1, self.walk_length):
                targets = np.concat([targets ,walks[:, i]])
                contexts = np.concat([contexts, walks[:, j]])
        targets = torch.tensor(targets,dtype=torch.long,device=self.device)
        contexts = torch.tensor(contexts,dtype=torch.long,device=self.device)
        return targets, contexts
    
    def train(self):
        for _ in range(self.sampling_times):
            # 生成随机游走序列
            walks = self.generate_walks()
            
            # 生成训练数据
            targets, contexts = self.generate_training_data(walks)
            dataset = torch.utils.data.TensorDataset(targets, contexts)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # 训练循环
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_targets, batch_contexts in tqdm(dataloader):
                    pos_scores = self.model(batch_targets, batch_contexts)
                    pos_labels = torch.ones_like(pos_scores)
                    
                    neg_targets = torch.randint(0, self.num_nodes, (batch_contexts.shape[0] * 3 // 4,), dtype=torch.long,device=self.device)
                    neg_contexts = torch.randint(0, self.num_nodes, (batch_contexts.shape[0] * 3 // 4,), dtype=torch.long,device=self.device)
                    neg_scores = self.model(neg_targets, neg_contexts)
                    neg_labels = torch.zeros_like(neg_scores)
                    
                    # 计算损失
                    x = torch.cat([pos_scores, neg_scores], dim=0)
                    x = F.sigmoid(x)
                    y = torch.cat([pos_labels, neg_labels], dim=0)
                    loss = self.loss_fn(x, y)
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
            self.model.normalize_embeddings()
            torch.save(self.model.get_embeddings().cpu(), "./Dataset/app_embeddings.pt")
        # 获取嵌入矩阵
        embeddings = self.model.get_embeddings().cpu()
        return embeddings
