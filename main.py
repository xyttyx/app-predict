import os

from utils import AppDataset, Trainer, Eval, load_poi_embedding
from Models import ModelAttention, ModelLSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split

from config import config_lstm
from config import config_attn

def main(config, seq_length:int|None = None, app_embedding_dim:int|None = None, user_embedding_dim:int|None = None, use_poi:bool|None = None):
    # 加载配置文件
    app_number = config.app_number
    user_number = config.user_number
    if app_embedding_dim == None:
        app_embedding_dim = config.app_embedding_dim
    if user_embedding_dim == None:
        user_embedding_dim = config.user_embedding_dim
    if seq_length == None:
        seq_length = config.seq_length
    train_epochs = config.train_epochs
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    model_name = config.model_name  
    Dataset_Path = config.Dataset_Path
    Save_Path = config.Save_Path
    if use_poi == None:
        use_poi = config.use_poi
    if not os.path.exists(Save_Path):
        os.makedirs(Save_Path)
    Model_Save_Path = os.path.join(Save_Path, "model")
    if not os.path.exists(Save_Path):
        os.makedirs(Save_Path)
    if not os.path.exists(Model_Save_Path):
        os.makedirs(Model_Save_Path)

    # 检查是否有可用的加速设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    if device.type == "cuda":
        print("Using GPU")
    elif device.type == "mps":
        print("Using Apple Silicon GPU")
    else:
        print("Using CPU")
    
    if use_poi:
        print("Using POI embedding")
    else:
        print("Not using POI embedding")
        
    # 加载数据集
    dataset = AppDataset(DatasetPath=Dataset_Path, length=seq_length)
    train_data, val_data = train_test_split(dataset.App_usage_trace, test_size=0.2, random_state=42)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    poi_embedding = load_poi_embedding(DatasetPath=Dataset_Path) if use_poi else None

    # 初始化模型
    if model_name == "lstm":
        print("Using LSTM model")
        model = ModelLSTM(
            app_number=app_number,
            user_number=user_number,
            app_embedding_dim=app_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            seq_length=seq_length,
            use_poi=use_poi,
            poi_embedding=poi_embedding,
        ).to(device)
    elif model_name == "attn":
        print("Using attention model")
        model = ModelAttention(
            app_number=app_number,
            user_number=user_number,
            app_embedding_dim=app_embedding_dim,
            user_embedding_dim=user_embedding_dim,
            seq_length=seq_length,
            use_poi=use_poi,
            poi_embedding=poi_embedding,
        ).to(device)

    # 加载模型
    if os.path.exists(os.path.join(Model_Save_Path, f"model_{model_name}_{"with" if use_poi else "without"}_poi_newest.pth")):
        model.load_state_dict(torch.load(os.path.join(Model_Save_Path, f"model_{model_name}_{"with" if use_poi else "without"}_poi_newest.pth"), weights_only=True))
        print("Model loaded successfully.")
    else:
        print("No model found, starting training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs, eta_min=learning_rate / 100)

    # 训练模型
    for epoch in range(1, train_epochs + 1):
        Trainer(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch
        )
        if False:
        # if epoch % (train_epochs // 3) == 0 :
            Eval(
                model=model,
                dataloader=val_dataloader,
                device=device,
            )
        scheduler.step()
        # 保存模型
        torch.save(model.state_dict(), os.path.join(Model_Save_Path, f"model_{model_name}_{"with" if use_poi else "without"}_poi_newest.pth"))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(Model_Save_Path, f"model_{model_name}_{"with" if use_poi else "without"}_poi_{epoch}.pth"))
    Eval(
        model=model,
        dataloader=val_dataloader,
        device=device,
    )
if __name__ == "__main__":
    main(config_attn, use_poi=False)
    main(config_lstm, use_poi=False)
    