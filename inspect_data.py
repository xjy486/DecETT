import pickle
import sys
import torch
import numpy as np

file_path = 'saved_data/v2ray_corr/data_bak.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded data type: {type(data)}")
    if isinstance(data, tuple):
        print(f"Tuple length: {len(data)}")
        for i, item in enumerate(data):
            print(f"Item {i} type: {type(item)}")
            if hasattr(item, 'shape'):
                print(f"Item {i} shape: {item.shape}")
            elif isinstance(item, list):
                print(f"Item {i} list length: {len(item)}")
                if len(item) > 0:
                    print(f"Item {i}[0] type: {type(item[0])}")
    else:
        print(f"Data: {data}")

except Exception as e:
    print(f"Error loading pickle: {e}")
