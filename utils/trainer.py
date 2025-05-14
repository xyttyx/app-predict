import os

import torch
from utils import metrics

from tqdm import tqdm

def Trainer(
        model,
        dataloader,
        optimizer,
        criterion,
        device,
        epoch
    ):
    running_loss = 0.0
    correct = 0
    prediction_total = []
    target_total = []
    # Train the model
    for item in tqdm(dataloader):
        x, target = item
        print("输入向量：")
        print(x[0])
        print("目标APP：", target[0])
        x = x.to(device)
        target = target.to(device)

        y = model(x)
        loss = criterion(y, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        prediction = torch.argmax(y, dim=1)
        prediction5 = y.topk(5)[1].T
        print("目标APP")
        print(target[0:10])
        print("Top 5 预测APP")
        torch.set_printoptions(threshold=float('inf'))
        print(prediction5[:, 0:10])
        prediction_total.append(prediction)
        target_total.append(target)
        
    accuracy = torch.sum(torch.cat(prediction_total) == torch.cat(target_total)).item() / len(torch.cat(target_total))
    loss = running_loss / len(dataloader)
    print(f"Train Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
        