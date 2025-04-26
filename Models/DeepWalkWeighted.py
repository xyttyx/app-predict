import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.data import Data
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
        
        # 初始化Skip-gram模型
        self.model = SkipGram(num_nodes, embedding_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.BCELoss()
    def get_neighbors(self, node: int) -> List[int]:
        mask = self.edge_index[0] == node
        if mask.sum() == 0:
            return None
        neighbors = self.edge_index[1][mask].tolist()
        weights = self.edge_weight[mask]
        weights = weights.sqrt()
        weights = weights / weights.sum()
        return random.choices(neighbors, weights=weights, k=1)[0]
    
    def _random_walk(self, start_node: int) -> List[int]:
        """从start_node开始随机游走"""
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur_node = walk[-1]
            neighbors = self.get_neighbors(cur_node)
            if not neighbors:
                break
            walk.append(neighbors)
        return walk
    
    def generate_walks(self) -> List[List[int]]:
        """为所有节点生成随机游走序列"""
        walks = []
        print("generate walks")
        for _ in tqdm(range(self.num_walks)):
            for node in range(self.num_nodes):
                walk_seq = self._random_walk(node)
                if len(walk_seq) == self.walk_length:
                    walks.append(walk_seq)
        return walks
    
    def generate_training_data(self, walks: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """从随机游走生成训练数据(目标词-上下文词对)"""
        pairs = []
        
        for walk in walks:
            length =  len(walk)
            middle = length // 2
            target = walk[middle]
            # 获取上下文窗口内的词
            for j in range(length):
                if target != walk[j]: 
                    pairs.append((target, walk[j]))
        
        # 转换为PyTorch张量
        targets = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        return targets, contexts
    
    def train(self) -> np.ndarray:
        """训练DeepWalk模型"""
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
                
                neg_targets = torch.randint(0, self.num_nodes, (batch_contexts.shape[0] * 3 // 4,), dtype=torch.long)
                neg_contexts = torch.randint(0, self.num_nodes, (batch_contexts.shape[0] * 3 // 4,), dtype=torch.long)
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
        
        # 获取嵌入矩阵
        embeddings = self.model.get_embeddings()
        return embeddings

# 示例用法
if __name__ == "__main__":
    # 创建一个简单的图 (无向)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
    ], dtype=torch.long)
    num_nodes = 6
    edge_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float)
    
    # 初始化DeepWalk
    deepwalk = DeepWalkWeighted(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        walk_length=5,
        num_walks=5,
        embedding_size=200,
        epochs=10,
        batch_size=32
    )
    
    # 训练并获取嵌入
    embeddings = deepwalk.train()
    print(f"\n生成的嵌入矩阵形状: {embeddings.shape}")
    print("前5个节点的嵌入示例:")
    print(embeddings[:5])