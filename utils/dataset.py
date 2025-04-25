import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class AppDataset(Dataset):
    def __init__(self, file_name=None, DatasetPath="Dataset", length=8):
        super(AppDataset, self).__init__()
        if file_name is None:
            #使用经过预处理的数据集，该数据集去除了使用记录过短或过长的用户，并对用户编号进行了重新映射
            file_name = "App_usage_trace_filtered.txt"
        App_usage_trace_path = os.path.join(DatasetPath, file_name)
        with open(App_usage_trace_path, "r") as f:
            App_usage_trace = f.readlines()
            App_usage_trace = [line.strip().split() for line in App_usage_trace]
            # 只保留用户编号、时间和应用编号, 顺序：user hour minute position app
            for i in range(len(App_usage_trace)):
                line = App_usage_trace[i]
                time = int(line[1]) % 1000000
                App_usage_trace[i] = torch.tensor((
                    int(line[0]),
                    time // 10000,
                    (time % 10000) // 200,
                    int(line[2]),
                    int(line[3])
                ))
            App_usage_trace = torch.stack(App_usage_trace)
            self.App_usage_trace_origin = App_usage_trace
            # 生成序列，每个序列的长度为length
            self.App_usage_trace = []
            for i in range(len(App_usage_trace) - (length + 1)):
                seq = App_usage_trace[i : i + (length + 1)]
                if seq[0][0] == seq[-1][0]:
                    self.App_usage_trace.append((seq[0:-1], seq[-1, 4])),
        print("App usage trace loaded, total number of sequences: ", len(self.App_usage_trace))
            
    def __len__(self):
        return len(self.App_usage_trace)
    
    def __getitem__(self, i):
        return self.App_usage_trace[i]

def load_poi_embedding(file_name=None, DatasetPath="Dataset"):
    if file_name is None:
        file_name = "base_poi.txt"
    poi_embedding_path = os.path.join(DatasetPath, file_name)
    with open(poi_embedding_path, "r") as f:
        poi_embedding = f.readlines()[1:]
        poi_embedding = [line.strip().split()[1:] for line in poi_embedding]
        poi_embedding = torch.tensor([[float(j) for j in i] for i in poi_embedding])
        poi_embedding = F.normalize(poi_embedding, dim=1)
    return poi_embedding  
