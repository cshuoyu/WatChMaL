'''
Author: Shuoyu Chen shuoyuchen.physics@gmail.com
Date: 2025-09-21 22:43:50
LastEditors: Shuoyu Chen shuoyuchen.physics@gmail.com
LastEditTime: 2025-10-05 11:42:15
FilePath: /schen/workspace/WatChMaL/inspect_h5.py
Description: 
'''
import h5py
import sys
import numpy as np

def print_h5_structure(name, obj, indent_level=0):
    """递归地打印出HDF5文件结构"""
    indent = "    " * indent_level
    if isinstance(obj, h5py.Group):
        print(f"{indent}📂 Group: {name}")
        for key, val in obj.items():
            print_h5_structure(key, val, indent_level + 1)
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
    else:
        print(f"{indent}❓ Unknown: {name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_h5.py <path_to_h5_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    
    print("-" * 60)
    print(f"Inspecting file: {file_path}")
    print("-" * 60)

    try:
        with h5py.File(file_path, 'r') as f:
            for key, val in f.items():
                print_h5_structure(key, val)
    except Exception as e:
        print(f"Error opening or reading file: {e}")
    
    print("-" * 60)