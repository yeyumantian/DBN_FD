# classical_ml.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 可选: XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# 可选: LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# 可选: CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


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


def get_baseline_models(random_state=0):
    """
    返回一个字典: {模型名: sklearn/catboost/lightgbm 模型实例}
    """
    models = {
        # 线性类
        "LogReg": LogisticRegression(
            max_iter=300,
            solver="lbfgs",
            multi_class="auto",
            n_jobs=-1
        ),
        "Ridge": RidgeClassifier(),

        # SVM 系列
        "LinearSVM": LinearSVC(),  # 线性核 SVM
        "SVM_RBF": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=False
        ),

        # 树模型系列
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        ),
        "GradBoost": GradientBoostingClassifier(
            random_state=random_state
        ),

        # 基础概率 & 距离模型
        "NaiveBayes": GaussianNB(),
        "KNN_5": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist"
        )

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )

    if HAS_CAT:
        # CatBoost 自带类别编码，这里我们当作普通数值使用
        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function='MultiClass',
            random_seed=random_state,
            verbose=False
        )

    return models


def run_baseline_experiments(feature_df, label_col="label", sorce_col=None, test_size=0.2, random_state=0, stratify=False):
    """
    用一批经典 ML 模型在给定特征上做 train/test，返回 results 字典，结构大致为:
    {
      model_name: {
         "model": pipeline模型,
         "accuracy": float,
         "f1_macro": float,
         "confusion_matrix": np.ndarray
      }, ...
    }

    sorce_col: 数据的来源列。如果不为空，基于此列进行分层划分。
    stratify: 是否使用分层划分，确保相同来源的样本不会同时出现在训练集和测试集。
    """
    X, y, feature_cols = build_X_y_from_feature_df(feature_df, label_col=label_col)

    if sorce_col is not None and stratify:
        print('不同来源的训练和测试')
        # 使用 sorce_col 进行分层，并确保数据来源在训练集和测试集之间不重叠
        # 我们根据 sorce_col 的唯一值进行划分
        unique_sources = feature_df[sorce_col].unique()
        train_sources, test_sources = train_test_split(unique_sources, test_size=test_size, random_state=random_state)
        print('训练集样本：',train_sources)
        print('测试集样本：',test_sources)
        # 通过源信息划分训练集和测试集
        train_mask = feature_df[sorce_col].isin(train_sources)
        test_mask = feature_df[sorce_col].isin(test_sources)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
    else:
        # 没有指定 sorce_col 或 stratify 为 False，普通划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )

    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"特征维度: {X_train.shape[1]}")
    print()

    models = get_baseline_models(random_state=random_state)
    results = {}

    for name, model in models.items():
        print("=" * 60)
        print(f"训练模型: {name}")

        # 注意: 树模型和 NaiveBayes 对标准化不敏感，但统一包个 Pipeline 简化代码
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        print(f"{name} - Accuracy: {acc:.4f}, F1_macro: {f1_macro:.4f}")
        print("分类报告:")
        print(classification_report(y_test, y_pred, digits=4))

        cm = confusion_matrix(y_test, y_pred)
        print("混淆矩阵:")
        print(cm)

        results[name] = {
            "model": pipe,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "confusion_matrix": cm,
            "feature_cols": feature_cols,
        }

    return results

#保存结果
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def save_all_result_df(results_df,path) :

    if not isinstance(results_df.index, pd.MultiIndex):
        raise TypeError("results_df 需要是 MultiIndex (env, model)")

    if results_df.index.names[0] != "env":
        raise ValueError(f"MultiIndex 第 0 级应为 'env'，当前为 {results_df.index.names[0]}")

    df = results_df.copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    print(f"[save_results_df] 结果已保存到：{path}")
    return df



#结果呈现部分========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def results_to_dataframe(results):
    """
    将 run_baseline_experiments 返回的 results 字典
    转换为一个整洁的 DataFrame（行=模型，列=指标）。
    """
    rows = []
    for model_name, info in results.items():
        rows.append({
            "model": model_name,
            "accuracy": info.get("accuracy", np.nan),
            "f1_macro": info.get("f1_macro", np.nan),
        })
    df = pd.DataFrame(rows).set_index("model")
    # 按 accuracy 排序
    df = df.sort_values(by="accuracy", ascending=False)
    return df
    
def results_env_to_dataframe(results_env):
    """
    results_env: 形如
      {
        env_name1: { model_name: {...}, ... },
        env_name2: { model_name: {...}, ... },
      }
    转换成一个 DataFrame:
      index = (env_name, model_name)
      columns = [accuracy, f1_macro]
    """
    rows = []
    for env_name, results in results_env.items():
        for model_name, info in results.items():
            rows.append({
                "env": env_name,
                "model": model_name,
                "accuracy": info.get("accuracy", np.nan),
                "f1_macro": info.get("f1_macro", np.nan),
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["env", "model"]).sort_values(
        by=["env", "accuracy"], ascending=[True, False]
    )
    return df

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_models_for_env(results_df: pd.DataFrame,
                        env: str,
                        metrics=("accuracy", "f1_macro"),
                        sort_by: str = "accuracy",
                        figsize=(7, 4),
                        dpi: int = 300,
                        save_path: str | Path | None = None):
    """
    在给定 env 下，画出所有模型的多个指标对比柱状图。
    默认画 accuracy 和 f1_macro。
    """
    if not isinstance(results_df.index, pd.MultiIndex):
        raise TypeError("results_df 需要是 MultiIndex (env, model)")

    # 取该 env 下所有模型
    sub = results_df.xs(env, level="env")
    # 只保留需要的指标
    sub = sub.loc[:, metrics].copy()
    sub["model"] = sub.index.astype(str)

    # 排序
    if sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=False)

    models = sub["model"].to_numpy()
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    width = 0.8 / len(metrics)  # 多指标并排柱宽自适应
    for i, m in enumerate(metrics):
        vals = sub[m].to_numpy()
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width,
               vals,
               width=width,
               label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Models performance @ env = '{env}'")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)

    # 让图更“科研感”一点
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot_models_for_env] 图已保存到：{save_path}")

    plt.show()


def plot_envs_for_model(results_df: pd.DataFrame,
                        model: str,
                        metrics=("accuracy", "f1_macro"),
                        sort_by: str = "accuracy",
                        figsize=(7, 4),
                        dpi: int = 300,
                        save_path: str | Path | None = None):
    """
    在给定 model 下，画出该模型在不同 env 上的表现对比。
    """
    if not isinstance(results_df.index, pd.MultiIndex):
        raise TypeError("results_df 需要是 MultiIndex (env, model)")

    # 取该 model 在所有 env 下的数据
    sub = results_df.xs(model, level="model")
    sub = sub.loc[:, metrics].copy()
    sub["env"] = sub.index.astype(str)

    if sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=False)

    envs = sub["env"].to_numpy()
    x = np.arange(len(envs))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    width = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        vals = sub[m].to_numpy()
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width,
               vals,
               width=width,
               label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=40, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Env comparison @ model = '{model}'")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot_envs_for_model] 图已保存到：{save_path}")

    plt.show()


