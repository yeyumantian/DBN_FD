import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd 
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def build_X_y_from_feature_df(feature_df, label_col="label"):
    """
    从特征 DataFrame 中抽取 X, y。
    只保留数值型特征列。
    """
    feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in feature_cols:
        feature_cols.remove(label_col)

    X = feature_df[feature_cols].values.astype(np.float32)
    y = feature_df[label_col].values.astype(np.int64)
    return X, y, feature_cols


class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid

        # 权重 & 偏置参数
        self.W = nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))

    def sample_from_p(self, p):
        # 对概率进行伯努利采样
        return torch.bernoulli(p)

    def v_to_h(self, v):
        # 由可视层到隐层的条件概率
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return p_h

    def h_to_v(self, h):
        # 由隐层到可视层的条件概率
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return p_v

    def forward(self, v):
        # 用于前向推理：给一个 v，返回隐层激活概率
        p_h = self.v_to_h(v)
        return p_h

    def contrastive_divergence(self, v0, lr=1e-3, k=1):
        """
        v0: [batch_size, n_vis]
        k : CD-k，这里默认 CD-1
        """
        v = v0
        # 正相（positive phase）
        p_h0 = self.v_to_h(v0)
        h0 = self.sample_from_p(p_h0)

        # Gibbs 采样
        for _ in range(k):
            p_v = self.h_to_v(h0)
            v = self.sample_from_p(p_v)
            p_h = self.v_to_h(v)
            h = self.sample_from_p(p_h)

        # 负相（negative phase）
        v_k = v
        p_hk = p_h

        # 更新梯度：dW ∝ v0^T h0 - v_k^T h_k
        positive_grad = torch.matmul(v0.t(), p_h0)
        negative_grad = torch.matmul(v_k.t(), p_hk)

        # 手动更新参数（最原始的写法，便于理解）
        self.W.data += lr * (positive_grad - negative_grad) / v0.size(0)
        self.v_bias.data += lr * torch.mean(v0 - v_k, dim=0)
        self.h_bias.data += lr * torch.mean(p_h0 - p_hk, dim=0)

        # 返回一个重构误差方便观察
        loss = torch.mean((v0 - v_k) ** 2)
        return loss.item()

class DBNClassifier(nn.Module):
    def __init__(self, layer_sizes, n_classes):
        """
        layer_sizes: [input_dim, hid1, hid2, ...]
        n_classes : 分类类别数
        """
        super(DBNClassifier, self).__init__()
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        # 创建若干个 RBM，用于无监督预训练
        self.rbms = nn.ModuleList([
            RBM(layer_sizes[i], layer_sizes[i+1])
            for i in range(self.n_layers)
        ])

        # 用同样的结构创建一个前馈神经网络，用于监督微调
        modules = []
        for i in range(self.n_layers):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            modules.append(nn.ReLU())
        # 最后一层分类头
        modules.append(nn.Linear(layer_sizes[-1], n_classes))
        self.classifier = nn.Sequential(*modules)

    def forward(self, x):
        return self.classifier(x)

    @torch.no_grad()
    def init_from_rbms(self):
        """
        用预训练好的 RBM 权重初始化前馈网络前几层。
        """
        linear_layers = [m for m in self.classifier.modules()
                         if isinstance(m, nn.Linear)]

        for i, rbm in enumerate(self.rbms):  # 最后一层 RBM 可选
            linear_layers[i].weight.data = rbm.W.data.t().clone()
            linear_layers[i].bias.data = rbm.h_bias.data.clone()

def pretrain_dbn(dbn, train_loader, n_epochs=5, lr=1e-3, device="gpu"):
    dbn.to(device)
    for layer_idx, rbm in enumerate(dbn.rbms):
        print(f"Pretraining RBM layer {layer_idx+1}/{dbn.n_layers}")
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_x, _ in train_loader:   # 无监督，不用标签
                v = batch_x.to(device)
                # 逐层传递：当前层的输入是前面所有 RBM 的隐层输出
                with torch.no_grad():
                    for prev_idx in range(layer_idx):
                        v = dbn.rbms[prev_idx].v_to_h(v)
                        v = (v > 0.5).float()  # 二值化一下

                loss = rbm.contrastive_divergence(v, lr=lr, k=1)
                epoch_loss += loss * v.size(0)
            print(f"  Epoch {epoch+1}/{n_epochs}, recon loss = {epoch_loss / len(train_loader.dataset):.6f}")
    # 预训练完成后，用 RBM 初始化前馈分类器权重
    dbn.init_from_rbms()

def finetune_dbn(dbn, train_loader, test_loader=None,
                 n_epochs=20, lr=1e-3, device="cpu"):
    dbn.to(device)
    optimizer = torch.optim.Adam(dbn.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(n_epochs):
        dbn.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = dbn(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f"[Fine-tune] Epoch {epoch+1}/{n_epochs}, "
              f"loss={train_loss:.4f}, acc={train_acc:.4f}")

        if test_loader is not None:
            acc, _ = evaluate_dbn(dbn, test_loader, device=device)
            test_accs.append(acc)

    return train_losses, train_accs, test_accs



def evaluate_dbn(dbn, data_loader, device="cuda"):
    dbn.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = dbn(batch_x)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"  [Eval] acc={acc:.4f}, f1_macro={f1:.4f}")
    return acc, f1

def plot_dbn_curves(train_losses, train_accs, test_accs=None,
                    save_path=None, figsize=(6,4), dpi=300):
    epochs = range(1, len(train_losses)+1)

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accs, label="Train acc", linestyle="--")
    if test_accs is not None and len(test_accs) == len(train_losses):
        ax2.plot(epochs, test_accs, label="Test acc", linestyle=":")
    ax2.set_ylabel("Accuracy")

    # 合并图例
    lines, labels = [], []
    for ax in [ax1, ax2]:
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="lower right", frameon=False)

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot_dbn_curves] 图已保存到: {save_path}")

    plt.show()
import matplotlib.pyplot as plt
from pathlib import Path




#主函数
def run_dbn_experiment(feature_df: pd.DataFrame, label_col: str = "label", 
                       sorce_col: str = None, stratify: bool = False, 
                       dbn_model=None, test_size=0.2, random_state=0,
                       n_pretrain_epochs=50, n_finetune_epochs=200, 
                       lr=1e-3, device="cuda"):
    """
    完整的 DBN 训练实验流程，包含数据读取、划分、预训练、微调及结果记录。
    返回的结果将被添加到结果字典中，更新对应的 'all_feature' 或 'all_feature_fc'。
    """
    # 1. 从 feature_df 中构建特征和标签
    X, y, feature_cols = build_X_y_from_feature_df(feature_df, label_col=label_col)
    
    # 2. 根据 sorce_col 划分训练集和测试集
    if sorce_col is not None and stratify:
        unique_sources = feature_df[sorce_col].unique()
        train_sources, test_sources = train_test_split(unique_sources, test_size=test_size, random_state=random_state)
        train_mask = feature_df[sorce_col].isin(train_sources)
        test_mask = feature_df[sorce_col].isin(test_sources)
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        print('训练集样本：',train_sources)
        print('测试集样本：',test_sources)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None)

    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"特征维度: {X_train.shape[1]}")
    # 3. 数据归一化
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)


    # 4. 将数据转换为 torch 张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 6. 定义 DBN 结构
    input_dim = X_train_t.shape[1]
    n_classes = len(np.unique(y_train))
    print("DBN input_dim =", input_dim, ", n_classes =", n_classes)

    layer_sizes = [input_dim, 256, 128]
    dbn = DBNClassifier(layer_sizes=layer_sizes, n_classes=n_classes)

    # 6) 预训练 + 微调

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrain_dbn(dbn, train_loader, n_epochs=n_pretrain_epochs, lr=lr, device=device)
    # 这里如果你想画曲线，可以让 finetune_dbn 返回 loss/acc，我下面给一个改造版本
    train_losses, train_accs, test_accs = finetune_dbn(
        dbn, train_loader, test_loader=test_loader,
        n_epochs=n_finetune_epochs, lr=lr, device=device
    )
    
   # 7) 最终测试性能
    test_acc, test_f1 = evaluate_dbn(dbn, test_loader, device=device)
    print(f"[DBN] final test_acc={test_acc:.4f}, test_f1={test_f1:.4f}")

    # 返回一个跟 baseline 兼容的结果 dict
    return {
        "DBN": {
            "accuracy": test_acc,
            "f1_macro": test_f1,
        }
    }, (train_losses, train_accs, test_accs)
 
