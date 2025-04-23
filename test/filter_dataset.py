import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime

DatasetPath = "Dataset"
file_name = "App_usage_trace_preprocessed.txt"

App_usage_trace_path = os.path.join(DatasetPath, file_name)
with open(App_usage_trace_path, "r") as f:
    App_usage_trace = f.readlines()
    App_usage_trace = [line.strip().split() for line in App_usage_trace]
    # 顺序：user time position app
    for i in range(len(App_usage_trace)):
        line = App_usage_trace[i]
        App_usage_trace[i] = (
            int(line[0]),
            int(line[1]),
            int(line[2]),
            int(line[3])
        )



filtered_App_usage_trace = []
for item in App_usage_trace:
    if len(filtered_App_usage_trace) == 0:
        filtered_App_usage_trace.append(item)   
        continue
    time = item[1] 
    time = datetime.datetime.strptime(str(time), f"%Y%m%d%H%M%S")
    previous_time = datetime.datetime.strptime(str(filtered_App_usage_trace[-1][1]), f"%Y%m%d%H%M%S")
    if time - previous_time > datetime.timedelta(minutes=1):
        filtered_App_usage_trace.append(item)
    else:
        if item[3] != filtered_App_usage_trace[-1][3]:
            filtered_App_usage_trace.append(item)
        else:
            continue

filtered_App_usage_trace_path = os.path.join(DatasetPath, "filtered_App_usage_trace.txt")
with open(filtered_App_usage_trace_path, "w") as f:
    for item in filtered_App_usage_trace:
        f.write(f"{item[0]} {item[1]} {item[2]} {item[3]}\n")
