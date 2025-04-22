import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelLSTM(nn.Module):
    def __init__(self,
            app_number,
            user_number,
            app_embedding_dim=200,
            user_embedding_dim=50,
            seq_length=8,
            num_layers=1,
            use_poi = False,
            poi_embedding = None,
        ):
        super(ModelLSTM, self).__init__()
        self.app_number = app_number
        self.user_number = user_number
        time_dim = 24 + 30 # 24小时 + 30分钟
        self.app_embedding = nn.Embedding(app_number, app_embedding_dim)
        self.user_embedding = nn.Embedding(user_number, user_embedding_dim)
        if use_poi:
            self.poi_embedding = nn.Embedding.from_pretrained(poi_embedding, freeze=True)
            self.lstm_input_dim = app_embedding_dim + time_dim + poi_embedding.size(1)
        else:
            self.lstm_input_dim = app_embedding_dim + time_dim
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_input_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_dim = self.lstm_input_dim + user_embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Tanh(),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Tanh(),
            nn.Linear(self.fc_dim, app_number),
        )

    def forward(self, x):
        apps = x[:, :, -1]
        users = x[:, 0, 0]
        hours = F.one_hot(x[:, :, 1], num_classes=24).float()
        minutes = F.one_hot(x[:, :, 2], num_classes=30).float()
        times = torch.cat([hours, minutes], dim=2)
        apps = self.app_embedding(apps)
        users = self.user_embedding(users)
        input = torch.cat([apps, times], dim=2)
        (h0, c0) = (torch.zeros(self.num_layers, apps.size(0), self.lstm_input_dim).to(apps.device),
                    torch.zeros(self.num_layers, apps.size(0), self.lstm_input_dim).to(apps.device))
        
        lstm_out, (hn, cn) = self.LSTM(input, (h0, c0))
        lstm_out = lstm_out[:, -1, :].squeeze(1)
        middle = torch.cat((lstm_out, users), dim=1)
        out = self.fc(middle)
        
        return out