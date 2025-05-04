import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelAttention(nn.Module):
    def __init__(self,
            app_number,
            user_number,
            app_embedding_dim=200,
            user_embedding_dim=50,
            time_embedding_dim=54,
            seq_length=8,
            use_poi = False,
            poi_embedding = None,
            poi_number = 9851,
            poi_embedding_dim=17,
        ):
        super(ModelAttention, self).__init__()
        self.app_number = app_number
        self.user_number = user_number
        self.app_embedding = nn.Embedding(app_number, app_embedding_dim)
        self.user_embedding = nn.Embedding(user_number, user_embedding_dim)
        self.time_embedding = nn.Embedding(240, time_embedding_dim)
        time_dim = 54
        attention_dim = app_embedding_dim + time_dim
        if use_poi:
            if poi_embedding is not None:
                poi_embedding = torch.tensor(poi_embedding)
            else:
                poi_embedding = torch.randn(poi_number, poi_embedding_dim)
            self.poi_embedding = nn.Embedding.from_pretrained(poi_embedding, freeze=True)
            attention_dim += poi_embedding.size(1)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=1, batch_first=True)
        self.weights = nn.Parameter(torch.ones(1, seq_length, 1) / seq_length, requires_grad=True)
        self.fc_dim = attention_dim + user_embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Tanh(),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Tanh(),
            nn.Linear(self.fc_dim, app_number),
        )

    def forward(self, x):
        # 顺序：user hour minute poi app
        apps = x[:, :, 3]
        times = x[:, :, 1]
        users = x[:, 0, 0]
        times = self.time_embedding(times)
        apps = self.app_embedding(apps)
        users = self.user_embedding(users)
        input = torch.cat([apps, times], dim=2)
        if hasattr(self, 'poi_embedding'):
            poi = x[:, :, 2]
            poi = self.poi_embedding(poi)
            input = torch.cat([input, poi], dim=2)
        attn_output, _ = self.attention(input, input, input)
        weight = F.softmax(self.weights, dim=1)
        attn_output = attn_output * weight
        attn_output = attn_output.sum(dim=1)
        middle = F.tanh(torch.cat((attn_output, users), dim=1))
        out = self.fc(middle)
        
        return out