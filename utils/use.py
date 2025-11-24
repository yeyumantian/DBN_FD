import pandas as pd
import scipy.io
import numpy as np

#读取mat文件===============================
def load_mat_and_check(path, target_var='variable_name'):
    # 加载mat文件
    mat_data = scipy.io.loadmat(path)
    # 打印变量名
    print("文件中的变量名：", list(mat_data.keys()))
    # 检查目标变量并输出信息
    if target_var in mat_data:
        var_val = mat_data[target_var]
        print(f"\n{target_var} 的值：\n{var_val}")
        print(f"{target_var} 的形状：{var_val.shape}")
        return var_val  # 可选：返回目标变量值
    print(f"\n未找到变量：{target_var}")
    return mat_data
#==============================================

#读入我保存的npz文件=========================
#X, y, label_strs, label_ids = load_npz_dataset("cond_xxx_win2048_step1024.npz")

def load_npz_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    label_strs = data["label_strs"]
    label_ids = data["label_ids"]
    ids = data['ids']
    print(X.shape, y.shape,ids.shape)
    print(label_strs)
    return X, y, label_strs, label_ids,ids
#==========================================