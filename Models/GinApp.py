import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData

num_apps = 1000  # 假设有1000个App
num_locs = 500   # 假设有500个位置
num_times = 24   # 假设有24个时间段

# 假设数据已预处理为以下格式
app_features = torch.randn(num_apps, 17)       # App类别特征 (17维)
loc_features = torch.randn(num_locs, 32)       # 位置特征 (POI+个性化时间分布)
time_features = torch.randn(num_times, 48)     # 时间特征 (48维)

# 边索引和权重 (格式: [source_nodes, target_nodes], edge_weights)
edge_index_app = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # App-App边
edge_weight_app = torch.tensor([0.8, 0.2])                         # 转移频率权重

edge_index_loc = torch.tensor([[0, 1]], dtype=torch.long)           # 位置-App边
edge_weight_loc = torch.tensor([0.5])

# 构建PyG图数据
data = HeteroData()
data['app'].x = app_features
data['loc'].x = loc_features
data['time'].x = time_features
data['app', 'to', 'app'].edge_index = edge_index_app
data['app', 'in', 'time'].edge_weight = edge_weight_app
data['app', 'at', 'loc'].edge_index = edge_index_loc

def weighted_random_walk(edge_index, edge_weight, start_node, walk_length):
    current_node = start_node
    walk = [current_node]
    for _ in range(walk_length):
        neighbors = edge_index[1, edge_index[0] == current_node]  # 获取邻居
        weights = edge_weight[edge_index[0] == current_node]       # 对应权重
        prob = weights / weights.sum()                            # 归一化概率
        next_node = neighbors[torch.multinomial(prob, 1)]         # 按概率采样
        walk.append(next_node)
        current_node = next_node
    return walk


class SameTypeAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),  # CAT操作后维度翻倍
            nn.ReLU()
        )

    def forward(self, target_feat, neighbor_feats):
        aggregated = neighbor_feats.sum(dim=0)               # SUM聚合邻居
        combined = torch.cat([target_feat, aggregated], dim=-1)  # CAT拼接
        return self.mlp(combined)                           # MLP映射
    
class CrossTypeAttention(nn.Module):
    def __init__(self, feat_dims):
        super().__init__()
        self.attention = nn.Linear(sum(feat_dims), 1)  # 注意力打分

    def forward(self, target_feat, feats_list):
        # feats_list: [f_app, f_loc, f_time]
        combined = torch.cat([target_feat.unsqueeze(0).expand(len(feats_list)), *feats_list], dim=-1)
        scores = torch.softmax(self.attention(combined), dim=0)  # 公式4
        return (scores * torch.stack(feats_list)).sum(dim=0)     # 公式3

class GinApp(nn.Module):
    def __init__(self, app_dim, loc_dim, time_dim, hidden_dim):
        super().__init__()
        self.app_agg = SameTypeAggregator(app_dim, hidden_dim)
        self.loc_agg = SameTypeAggregator(loc_dim, hidden_dim)
        self.time_agg = SameTypeAggregator(time_dim, hidden_dim)
        self.attention = CrossTypeAttention([hidden_dim, hidden_dim, hidden_dim])

    def forward(self, data, target_node):
        # 1. 采样邻居
        app_neighbors = weighted_random_walk(data.edge_index_app, data.edge_weight_app, target_node, 5)
        loc_neighbors = weighted_random_walk(data.edge_index_loc, data.edge_weight_loc, target_node, 5)
        
        # 2. 同类型聚合
        f_app = self.app_agg(data.x_app[target_node], data.x_app[app_neighbors])
        f_loc = self.loc_agg(data.x_loc[target_node], data.x_loc[loc_neighbors])
        
        # 3. 跨类型注意力聚合
        f_final = self.attention(data.x_app[target_node], [f_app, f_loc, data.x_time[target_node]])
        return f_final
    
# 示例使用
def train(model, data, optimizer):
    model.train()
    pos_pairs = [(0, 1), (1, 2)]  # 正样本对（共现节点）
    neg_samples = [3, 4]           # 负样本
    
    for v_i, v_j_p in pos_pairs:
        optimizer.zero_grad()
        h_i = model(data, v_i)
        h_j_p = model(data, v_j_p)
        pos_loss = -torch.log(torch.sigmoid(h_i.dot(h_j_p)))  # 正样本损失
        
        neg_loss = 0
        for v_j_n in neg_samples:
            h_j_n = model(data, v_j_n)
            neg_loss += -torch.log(1 - torch.sigmoid(h_i.dot(h_j_n)))  # 负样本损失
        
        loss = pos_loss + neg_loss / len(neg_samples)
        loss.backward()
        optimizer.step()

def predict_next_app(model, data, current_app, current_loc, current_time, k=5):
    h_app = model(data, current_app)
    h_loc = model(data, current_loc)
    h_time = model(data, current_time)
    
    # 计算所有候选App的得分
    scores = []
    for app in range(data.x_app.size(0)):
        h_candidate = model(data, app)
        score = torch.cosine_similarity(h_app + h_loc + h_time, h_candidate, dim=0)
        scores.append((app, score))
    
    # 返回Top-K预测
    return sorted(scores, key=lambda x: -x[1])[:k]

def main():
    # 初始化模型和优化器
    model = GinApp(app_dim=17, loc_dim=32, time_dim=48, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    for epoch in range(100):
        train(model, data, optimizer)
    
    # 预测下一个App
    current_app = 0
    current_loc = 1
    current_time = 2
    predictions = predict_next_app(model, data, current_app, current_loc, current_time)
    
    print("Top-K Predictions:", predictions)
if __name__ == "__main__":
    main()