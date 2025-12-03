import numpy as np
import sys

# --- 配置 ---
# 请确认这是您刚刚创建的、包含816个坐标的LUT文件
LUT_FILE_PATH = '/home/schen/mpmt_lut_816.npz'

def print_lut_contents(file_path):
    """
    加载并完整打印 .npz 文件中的 'pmt_module_positions' 数组。
    """
    print("-" * 60)
    print(f"Reading and printing contents of: {file_path}")
    print("-" * 60)

    try:
        # 加载文件
        data = np.load(file_path)
        
        # 确认关键数组是否存在
        if 'pmt_module_positions' not in data:
            print("❌ ERROR: Array 'pmt_module_positions' not found in the file.")
            return

        positions = data['pmt_module_positions']
        
        # 打印摘要信息
        print(f"Array name: 'pmt_module_positions'")
        print(f"Shape: {positions.shape}")
        print(f"Data type: {positions.dtype}\n")

        # 设置Numpy打印选项，以确保打印所有行，而不是用"..."省略
        np.set_printoptions(threshold=sys.maxsize)
        
        # 打印整个数组
        print("Full (816, 2) Coordinate Array:")
        print(positions)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("-" * 60)
        print("Printing complete.")
        print("-" * 60)

if __name__ == '__main__':
    print_lut_contents(LUT_FILE_PATH)