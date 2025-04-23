import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import AppDataset

# 测试数据集使用“直接输出上一个应用作为预测值”策略的accuracy
# 准确率为 60.13%
# 基于此数据，对数据集进行过滤，参见 test/filter_dataset.py
# 过滤后的数据集使用此策略的准确率为 11.51%
dataset = AppDataset(DatasetPath="Dataset", length=2)

count = 0
for i in range(len(dataset.App_usage_trace)):
    if dataset.App_usage_trace[i][0][0, -1] == dataset.App_usage_trace[i][0][1, -1]:
        count += 1

print(count / len(dataset.App_usage_trace))
