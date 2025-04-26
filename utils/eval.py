import os
import torch
from utils import metrics

from tqdm import tqdm

def Eval(
    model,
    dataloader,
    device,
    k=1
):
    with torch.no_grad():
        y_total = []
        target_total = []
        for item in tqdm(dataloader):
            x, target = item
            x = x.to(device)
            target = target.to(device)

            y = model(x)
            y_total.append(y)
            target_total.append(target)

        # Calculate metrics
        y_total = torch.cat(y_total, dim=0).cpu().numpy()
        target_total = torch.cat(target_total, dim=0).cpu().numpy()
        metrics_data = metrics(target_total, y_total, n_classes=model.app_number,k=k)

        # Print the metrics
        info = f"Eval:\n"
        for key,value in metrics_data.items() :
            info += f"Acc@{key}: {value:.4f} | "
        print(info)