import os
import torch
from utils import metrics

from tqdm import tqdm

def Eval(
    model,
    dataloader,
    device,
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
        y_total = torch.stack(y_total).cpu().numpy()
        target_total = torch.stack(target_total).cpu().numpy()
        metrics_data = metrics(target_total, y_total, n_classes=model.app_number)

        # Print the metrics
        print(
            f"Eval:\n"+
            f"Acc@1: {metrics_data['accuracy@1']:.4f} | "+
            f"Acc@2: {metrics_data['accuracy@2']:.4f} | "+
            f"Acc@3: {metrics_data['accuracy@3']:.4f} | "+
            f"Acc@4: {metrics_data['accuracy@4']:.4f} | "+
            f"Acc@5: {metrics_data['accuracy@5']:.4f}"
        )