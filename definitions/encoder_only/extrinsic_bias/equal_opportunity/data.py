import json
import pandas as pd

def load_classification_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_instances_by_group(data, label_key='label', group_key='gender'):
    grouped = {}
    for entry in data:
        label = entry[label_key]
        group = entry[group_key]
        pred = entry["prediction"]
        if (group, label) not in grouped:
            grouped[(group, label)] = {"TP": 0, "Total": 0}
        if label == pred:
            grouped[(group, label)]["TP"] += 1
        grouped[(group, label)]["Total"] += 1
    return grouped
