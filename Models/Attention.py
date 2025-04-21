import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelAttention(nn.Module):
    def __init__(self,
            app_number,
            user_number,
            app_embedding_dim=200,
            user_embedding_dim=50,
            seq_length=8,
        ):
        super(ModelAttention, self).__init__()
        self.app_number = app_number
        self.user_number = user_number
        self.app_embedding = nn.Embedding(app_number, app_embedding_dim)
        self.user_embedding = nn.Embedding(user_number, user_embedding_dim)
        time_dim = 24 + 30 # 24小时 + 30分钟
        self.attention = nn.MultiheadAttention(app_embedding_dim + time_dim, num_heads=1, batch_first=True)
        self.fc_dim = app_embedding_dim + user_embedding_dim + time_dim
        self.fc = nn.Sequential(
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Tanh(),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Tanh(),
            nn.Linear(self.fc_dim, app_number),
        )

    def forward(self, x):
        # 顺序：user hour minute app
        apps = x[:, :, -1]
        users = x[:, 0, 0]
        hours = F.one_hot(x[:, :, 1], num_classes=24).float()
        minutes = F.one_hot(x[:, :, 2], num_classes=30).float()
        times = torch.cat([hours, minutes], dim=2)
        apps = self.app_embedding(apps)
        users = self.user_embedding(users)
        input = torch.cat([apps, times], dim=2)
        attn_output, attn_output_weights = self.attention(input, input, input)
        attn_output = attn_output.mean(dim=1).squeeze(1)
        middle = F.tanh(torch.cat((attn_output, users), dim=1))
        out = self.fc(middle)
        
        return out