import os
import scipy.io
import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
notebook_path = Path(sys.argv[0]).parent if "ipykernel" in sys.argv[0] else Path.cwd()
sys.path.append(str(notebook_path))
from utils import use  

def sliding_windows(signal, window_size=2048, step_size=1024):
    """
    对一条 1D 信号做滑动窗口切片。
    返回: (windows, starts)
      - windows: list[np.ndarray]，每个长度=window_size
      - starts:  list[int]，每段的起始下标
    末尾不足一个完整窗口的部分直接丢弃。
    """
    sig = np.asarray(signal, dtype=np.float32)
    L = len(sig)
    if L < window_size:
        return [], []
    windows = []
    starts = []
    for start in range(0, L - window_size + 1, step_size):
        end = start + window_size
        windows.append(sig[start:end])
        starts.append(start)
    return windows, starts

def expand_df_with_sliding_windows(df, window_size=2048, step_size=1024):
    """
    输入: 原始 df（包含列 '数据'）
    输出: 扩充后的 expanded_df
    """
    rows = []
    for idx, row in df.iterrows():
        signal = row['数据']
        windows, starts = sliding_windows(signal, window_size, step_size)

        for w, s in zip(windows, starts):
            new_row = row.to_dict()  # 拷贝原始元信息
            new_row['数据'] = w       # 用片段替换原数据
            new_row['segment_start'] = s
            new_row['segment_end'] = s + len(w)
            new_row['segment_id'] = f"{idx}_{s}"
            rows.append(new_row)

    expanded_df = pd.DataFrame(rows)
    expanded_df['数据长度'] = expanded_df['数据'].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
    return expanded_df


def get_fault_label(df,fault_columns,is_print = False):
    # 获取所有标签的有序列表（按字典序）
    unique_labels = sorted(df[fault_columns].unique())

    # 构造映射: 标签字符串 -> 整数 id
    label2id = {lab: i for i, lab in enumerate(unique_labels)}

    # 加一列 数字标签
    df['label_id'] = df[fault_columns].map(label2id)

    if is_print == True:
        # 数字 id -> 原始标签 对照分布
        print("===== 扩充后 标签分布（带中文标签） =====")
        label_count = df[fault_columns].value_counts()
        for lab in unique_labels:
            print(f"id={label2id[lab]:2d}, 标签='{lab}': {label_count[lab]} 段样本")

        if '载荷' in df.columns:
            print("===== 扩充后 载荷 × 故障标签 交叉表 =====")
            print(pd.crosstab(df['载荷'], df[fault_columns]))
            print()

        print("=====  df 基本信息 =====")
        print(df.info())
        print()

        # 每条原始信号的长度描述
        print("===== 信号长度统计 df['数据长度'] =====")
        print(df['数据长度'].describe())
        print()

        # 各类标签分布（故障标签）
        print("===== 样本 故障标签 分布 =====")
        print(df[fault_columns].value_counts())
        print()

        # 如果你有 “故障类型”（内圈/外圈/滚动体/正常）
        if '故障类型' in df.columns:
            print("===== 样本 故障类型 分布 =====")
            print(df['故障类型'].value_counts())
            print()

        # 载荷 / rpm 分布（看工况的多样性）
        if '载荷' in df.columns:
            print("===== 样本 各载荷下样本数 =====")
            print(df.groupby('载荷').size())
            print()

        # 载荷 / rpm 分布（看工况的多样性）
        if '采样频率下' in df.columns:
            print("===== 样本 各采样频率下样本数 =====")
            print(df.groupby('采样频率').size())
            print()

        if 'rpm' in df.columns:
            print("===== 样本 各转速下样本数 =====")
            print(df.groupby('rpm').size())
            print()

    return df,unique_labels
    
def filter_conditions(df, config):
    """
    df: 原始清洗后的 df
    config: 一个字典，包含筛选选项（值为列表或 None）
    返回: filtered_df, condition_name
    """
    filtered = df.copy()
    name_parts = []

    def apply_filter(column, key):
        nonlocal filtered, name_parts
        values = config.get(key)
        if values:  # 不为空或 None 才筛选
            filtered = filtered[filtered[column].isin(values)]
            name_parts.append(f"{key}_" + "-".join(map(str, values)))

    # 字段名映射（让 key 更统一）
    mapping = {
        "sampling_rates": ("采样频率",),
        "sampling_positions": ("采样位置",),
        "bearing_positions": ("轴承位置",),
        "loads": ("载荷",),
        "rpms": ("rpm",),
        "fault_types": ("故障类型",),
        "or_positons": ("or采样位置",),
    }

    for key, (column,) in mapping.items():
        apply_filter(column, key)

    filtered = filtered.reset_index(drop=True)

    # 构建简单的名字
    if name_parts:
        cond_name = "cond_" + "_".join(name_parts)
    else:
        cond_name = "cond_all"
    print("筛选后样本数：", len(filtered))
    print("条件名称：", cond_name)

    return filtered, cond_name

def save_npz_data(expanded_df, unique_labels, cond_name,
                  window_size, step_size,
                  save_dir):
    """
    expanded_df: 滑窗+标签编码后的 df（必须包含 '数据' 和 'label_id'）
    unique_labels: 按 id 顺序排列的标签名字列表（get_fault_label 返回的）
    cond_name: 工况名称（filter_conditions 返回的）
    window_size: 滑窗长度（外部配置的 WINDOW_SIZE）
    step_size: 步长（STEP_SIZE）
    save_dir: 保存目录
    """
    
    signals = expanded_df['数据'].values
    labels  = expanded_df['label_id'].values
    
    ids = expanded_df['编号'].values  # 假设 '编号' 是你需要保存的字段

    X = np.stack(signals, axis=0).astype(np.float32)  # (N, window_size)
    y = labels.astype(np.int64)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    label_strs = np.array(unique_labels)         # id 对应标签字符串
    label_ids = np.arange(len(unique_labels))    # [0,1,2,...]

    filename = f"{cond_name}_win{window_size}_step{step_size}.npz"
    out_path = os.path.join(save_dir, filename)
    np.savez(out_path,
             X=X,
             y=y,
             label_strs=label_strs,
             label_ids=label_ids,
             ids=ids)
    print(f"npz 文件已保存到: {out_path}")
    return out_path

