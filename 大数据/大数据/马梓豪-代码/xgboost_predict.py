# -*- coding: utf-8 -*-
"""
good_rating_model_final_v2.py

功能总览
- 读入 predict_data.csv，目标列 good_rating。
- 先全局剔除 good_rating==100。
- 分类通道：按固定阈值分箱 [0,90), [90,95), [95,99)，若某类样本不足则自动与相邻类合并（直到每类>=MIN_CLASS_COUNT）。
  若仍不足两类 -> 自动回退为回归。
- 回归通道：对 y 做 logit 变换训练 (0–100→(0,1)→logit)，预测后反变换回 0–100。
  同时对比 XGBoost/HistGB + logit、HuberRegressor、QuantileRegressor(τ=0.5)。
- CV 可切换：'stratified'（默认）/ 'group_city'（GroupKFold）/ 'time_entry_date'（TimeSeriesSplit）。
- 特征工程：OneHot(title, city)，entry_date → year/month/day/dow；排除 reply_*_score 三列；默认不使用 name。
- 兼容新旧 sklearn 的 OneHotEncoder（sparse_output / sparse）。
- 所有图像保存到当前目录：cv_curve.png / confusion_matrix.png / roc_curve.png /
  pred_vs_actual.png / feature_importance_top20.png / permutation_importance_top15.png
- 产物保存到 artifacts/：best_model_*.pkl、cv_results.csv、feature_importances.csv
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ===== 字体与显示 =====
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 模型依赖（优先 XGBoost，不可用回退 HistGB） =====
use_xgb = True
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    use_xgb = False
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GroupKFold, TimeSeriesSplit, KFold, GridSearchCV
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, RocCurveDisplay, roc_auc_score,
    r2_score, mean_absolute_error
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor, QuantileRegressor
import joblib

# ========== 配置 ==========
DATA_PATH = "predict_data.csv"
TARGET = "good_rating"
DROP_COLS = {"reply_quality_score", "service_attitude_score", "reply_speed_score"}
USE_NAME = False   # 高基数，默认不使用

# 固定阈值分箱（左闭右开）；我们已剔除100，最右端[99,100)通常为空
RULE_BINS = [0, 90, 95, 99, 100]
MIN_CLASS_COUNT = 25  # 每类最少样本数（可根据数据量调整）

# 回归(logit) 参数
Y_MIN, Y_MAX = 0.0, 100.0
EPS = 1e-3  # 防止 p=0/1

# CV 模式: 'stratified' | 'group_city' | 'time_entry_date'
CV_MODE = "stratified"
N_SPLITS = 5

# ========== 工具 ==========
def savefig(path):
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

def to_logit(y):
    y = np.asarray(y, dtype=float).ravel()
    p = (y - Y_MIN) / (Y_MAX - Y_MIN)
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def from_logit(z):
    z = np.asarray(z, dtype=float).ravel()
    p = 1.0 / (1.0 + np.exp(-z))
    y = p * (Y_MAX - Y_MIN) + Y_MIN
    return y

def merge_sparse_bins(y_cat: pd.Series, min_count: int) -> pd.Series:
    """
    自动合并稀疏类：把样本最少的区间与其相邻区间合并，直到每类样本数 >= min_count 或只剩 1 类。
    输入可为 Categorical/字符串 Series；返回字符串 Series。
    """
    s = y_cat if isinstance(y_cat, pd.Series) else pd.Series(y_cat)
    s = s.astype(str)  # 统一字符串
    cats = list(pd.Categorical(s).categories)  # 当前类别顺序

    def _counts(series, cats_order):
        vc = series.value_counts()
        return pd.Series([vc.get(c, 0) for c in cats_order], index=cats_order)

    if len(cats) <= 1:
        return s

    while True:
        vc = _counts(s, cats)
        if len(vc) <= 1 or vc.min() >= min_count:
            break

        idx_min = int(np.argmin(vc.values))
        if idx_min == 0:
            neighbor = 1
        elif idx_min == len(vc) - 1:
            neighbor = len(vc) - 2
        else:
            neighbor = idx_min - 1 if vc.iloc[idx_min - 1] >= vc.iloc[idx_min + 1] else idx_min + 1

        src = cats[idx_min]; dst = cats[neighbor]
        s = s.replace(src, dst)  # 合并标签
        cats.pop(idx_min)        # 删除源类别，保持顺序

    return s

def build_cv_for_training(df_train, mode="stratified", n_splits=5, entry_date_col="entry_date"):
    """
    返回 (cv, groups)
    - stratified: StratifiedKFold, groups=None
    - group_city: GroupKFold, groups=array(city)
    - time_entry_date: TimeSeriesSplit, groups=None
    """
    if mode == "group_city":
        if df_train is None or "city" not in df_train.columns:
            print("⚠️ 未找到 city 列，回退 StratifiedKFold")
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None
        groups = df_train["city"].astype(str).fillna("missing").values
        return GroupKFold(n_splits=n_splits), groups
    if mode == "time_entry_date":
        if df_train is None or entry_date_col not in df_train.columns:
            print("⚠️ 未找到 entry_date，回退 StratifiedKFold")
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None
        return TimeSeriesSplit(n_splits=n_splits), None
    # 默认分层
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None

# ========== 读取数据 ==========
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"未找到数据文件：{DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("数据形状：", df.shape)

if TARGET not in df.columns:
    raise ValueError("未找到目标列 good_rating。")
df = df.dropna(subset=[TARGET]).copy()
if df[TARGET].dtype == object:
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()

# 剔除 100
before = df.shape[0]
df = df[df[TARGET] != 100].copy()
print(f"已剔除 good_rating==100 的样本：{before - df.shape[0]} 条；当前样本量：{df.shape[0]}")
if df.empty:
    raise ValueError("剔除 good_rating==100 后无数据。")

# ========== 特征工程 ==========
feature_cols = [c for c in df.columns if c not in {TARGET} | DROP_COLS]
if not USE_NAME and "name" in feature_cols:
    feature_cols.remove("name")

# 解析 entry_date
date_col = None
for cand in ["entry_date", "entry_data", "enter_date", "entry_dt"]:
    if cand in feature_cols:
        date_col = cand
        break

if date_col is not None:
    parsed = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    if parsed.notna().mean() > 0.5:
        df[date_col + "_year"]  = parsed.dt.year
        df[date_col + "_month"] = parsed.dt.month
        df[date_col + "_day"]   = parsed.dt.day
        df[date_col + "_dow"]   = parsed.dt.dayofweek
        feature_cols.remove(date_col)
        feature_cols += [date_col + "_year", date_col + "_month", date_col + "_day", date_col + "_dow"]

X_all = df[feature_cols].copy()
y_raw = df[TARGET].copy()

# 自动识别类别/数值；确保 title, city 独热
special_cats = [c for c in ["title", "city"] if c in feature_cols]
cat_cols, num_cols = [], []
for c in X_all.columns:
    if c in special_cats:
        cat_cols.append(c)
    elif X_all[c].dtype == object or str(X_all[c].dtype).startswith("category"):
        cat_cols.append(c)
    else:
        num_cols.append(c)

print(f"数值特征数：{len(num_cols)}，类别特征数：{len(cat_cols)}")

# OneHot 兼容
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)         # <1.2

numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  ohe),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols)
    ],
    remainder="drop"
)

# ========== 优先：固定阈值分箱 + 自动合并 ==========
y_bins = pd.cut(y_raw, bins=RULE_BINS, right=False, include_lowest=True)
y_bins = pd.Series(y_bins.astype('category'))
present_cats = [c for c in y_bins.cat.categories if (y_bins == c).sum() > 0]
y_bins = y_bins.cat.set_categories(present_cats, ordered=True)
y_bins_merged = merge_sparse_bins(y_bins, MIN_CLASS_COUNT)

TASK_MODE = "classification" if y_bins_merged.nunique() >= 2 else "regression"

# ======================================
# ============= 分类通道 ===============
# ======================================
if TASK_MODE == "classification":
    # 原始区间标签（字符串）
    y_cls = y_bins_merged.astype(str)

    # —— 标签编码（关键补丁）：y -> 数字标签 0..K-1 ——
    le = LabelEncoder()
    y_codes = pd.Series(le.fit_transform(y_cls), index=y_cls.index)  # 数字标签
    class_names = list(le.classes_)  # 原始区间标签顺序

    # 时间 CV 时的排序（若 entry_date 存在）
    if "entry_date" in df.columns:
        entry_for_cv = pd.to_datetime(df["entry_date"], errors="coerce")
    else:
        entry_for_cv = pd.Series(pd.NaT, index=df.index)

    if CV_MODE == "time_entry_date" and entry_for_cv.notna().sum() > 0:
        order = np.argsort(entry_for_cv.fillna(entry_for_cv.min()))
        X_all   = X_all.iloc[order].reset_index(drop=True)
        y_codes = y_codes.iloc[order].reset_index(drop=True)
        y_cls   = y_cls.iloc[order].reset_index(drop=True)  # 仅用于报告还原
        df      = df.iloc[order].reset_index(drop=True)

    # 切分（分层用数字标签）
    X_train, X_test, y_train_num, y_test_num = train_test_split(
        X_all, y_codes, test_size=0.2, random_state=42, stratify=y_codes
    )
    df_train = df.loc[y_train_num.index]

    # 构建 CV（返回 cv 与 groups，如果是 GroupKFold 则 groups 不为 None）
    cv, groups = build_cv_for_training(
        df_train, mode=CV_MODE, n_splits=N_SPLITS, entry_date_col="entry_date"
    )

    # 模型与参数网格
    if use_xgb:
        clf = XGBClassifier(
            objective="multi:softprob" if len(class_names) > 2 else "binary:logistic",
            num_class=None if len(class_names) <= 2 else len(class_names),
            eval_metric="mlogloss" if len(class_names) > 2 else "auc",
            tree_method="hist",
            random_state=42, n_estimators=300, n_jobs=4
        )
        param_grid = {
            "model__n_estimators": [300, 500],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8, 1.0],
        }
    else:
        clf = HistGradientBoostingClassifier(random_state=42)
        param_grid = {
            "model__max_depth": [None, 8],
            "model__max_leaf_nodes": [31, 63, 127],
            "model__learning_rate": [0.05, 0.1],
        }

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
    scoring = "f1_macro" if len(class_names) > 2 else "f1"

    grid = GridSearchCV(
        estimator=pipe, param_grid=param_grid, scoring=scoring,
        cv=cv, n_jobs=4, verbose=1, return_train_score=True
    )

    print("\n开始网格搜索（分类）...")
    # 若是 GroupKFold，需要在 fit 时传入 groups 参数
    fit_kwargs = {}
    if isinstance(cv, GroupKFold):
        fit_kwargs["groups"] = df_train["city"].astype(str).fillna("missing").values
    grid.fit(X_train, y_train_num, **fit_kwargs)
    print("最佳参数：", grid.best_params_)
    print("最佳CV得分：", grid.best_score_)

    # CV 曲线
    cv_results = pd.DataFrame(grid.cv_results_)
    plt.figure()
    plt.plot(range(len(cv_results)), cv_results["mean_test_score"], marker="o")
    plt.title("交叉验证：平均验证分数（按参数组合索引）")
    plt.xlabel("参数组合索引"); plt.ylabel("平均验证分数"); plt.grid(True); plt.tight_layout()
    savefig("cv_curve.png")

    # 测试集评估：先预测数字标签，再映射回原始区间标签
    best_model   = grid.best_estimator_
    y_pred_num   = best_model.predict(X_test)
    y_true_label = le.inverse_transform(np.asarray(y_test_num))
    y_pred_label = le.inverse_transform(np.asarray(y_pred_num))

    print("\n===== 测试集分类报告 =====")
    print(classification_report(y_true_label, y_pred_label,
                                labels=class_names, target_names=class_names, zero_division=0))

    metrics = {
        "accuracy": accuracy_score(y_true_label, y_pred_label),
        "precision_macro": precision_score(y_true_label, y_pred_label, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_label, y_pred_label, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_label, y_pred_label, average="macro", zero_division=0),
    }
    print("===== 指标汇总 =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 混淆矩阵（原始区间标签顺序）
    cm = confusion_matrix(y_true_label, y_pred_label, labels=class_names)
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("混淆矩阵"); plt.xlabel("预测类别"); plt.ylabel("真实类别"); plt.colorbar(im)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.tight_layout()
    savefig("confusion_matrix.png")

    # 二分类 + XGB：ROC
    if (len(class_names) == 2) and use_xgb:
        try:
            RocCurveDisplay.from_estimator(best_model, X_test, y_test_num)
            plt.title("ROC 曲线（测试集）"); plt.tight_layout()
            savefig("roc_curve.png")
            y_proba = best_model.predict_proba(X_test)[:, 1]
            print("测试集 AUC：", roc_auc_score(y_test_num, y_proba))
        except Exception as e:
            print("绘制ROC失败：", e)

# ======================================
# ============= 回归通道 ===============
# ======================================
else:
    # 按时间 CV 需要排序
    if "entry_date" in df.columns:
        entry_for_cv = pd.to_datetime(df["entry_date"], errors="coerce")
    else:
        entry_for_cv = pd.Series(pd.NaT, index=df.index)

    if CV_MODE == "time_entry_date" and entry_for_cv.notna().sum() > 0:
        order = np.argsort(entry_for_cv.fillna(entry_for_cv.min()))
        X_all = X_all.iloc[order].reset_index(drop=True)
        y_raw = y_raw.iloc[order].reset_index(drop=True)
        df = df.iloc[order].reset_index(drop=True)

    # 1) XGB/HistGB + logit（自定义包装器）
    if use_xgb:
        base_reg = XGBRegressor(tree_method="hist", random_state=42, n_estimators=400, n_jobs=4)
        param_grid = {
            "model__regressor__n_estimators": [300, 600],
            "model__regressor__max_depth": [4, 6],
            "model__regressor__learning_rate": [0.05, 0.1],
            "model__regressor__subsample": [0.8, 1.0],
            "model__regressor__colsample_bytree": [0.8, 1.0],
        }
    else:
        base_reg = HistGradientBoostingRegressor(random_state=42)
        param_grid = {
            "model__regressor__max_depth": [None, 8],
            "model__regressor__max_leaf_nodes": [31, 63, 127],
            "model__regressor__learning_rate": [0.05, 0.1],
        }

    class LogitWrap:
        def __init__(self, regressor):
            self.regressor = regressor
        def set_params(self, **params):
            for k, v in params.items():
                if k.startswith("regressor__"):
                    setattr(self.regressor, k.split("__", 1)[1], v)
            return self
        def get_params(self, deep=True):
            params = {}
            if hasattr(self.regressor, "get_params"):
                for k, v in self.regressor.get_params(deep=deep).items():
                    params["regressor__" + k] = v
            return params
        def fit(self, X, y):
            z = to_logit(y)
            self.regressor.fit(X, z)
            return self
        def predict(self, X):
            zhat = self.regressor.predict(X)
            return from_logit(zhat)

    model = LogitWrap(base_reg)
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    # CV
    if CV_MODE == "group_city" and "city" in df.columns:
        groups_all = df["city"].astype(str).fillna("missing").values
        cv = list(GroupKFold(n_splits=N_SPLITS).split(X_all, y_raw, groups=groups_all))
    elif CV_MODE == "time_entry_date" and entry_for_cv.notna().sum() > 0:
        cv = TimeSeriesSplit(n_splits=N_SPLITS)
    else:
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    scoring = "neg_root_mean_squared_error"

    grid = GridSearchCV(
        estimator=pipe, param_grid=param_grid, scoring=scoring,
        cv=cv, n_jobs=4, verbose=1, return_train_score=True
    )
    print("\n开始网格搜索（回归, logit 目标）...")
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_raw, test_size=0.2, random_state=42)
    grid.fit(X_train, y_train)
    print("最佳参数：", grid.best_params_)
    print("最佳CV得分：", grid.best_score_)  # 负的 RMSE

    # CV 曲线
    cv_results = pd.DataFrame(grid.cv_results_)
    plt.figure()
    plt.plot(range(len(cv_results)), cv_results["mean_test_score"], marker="o")
    plt.title("交叉验证：平均验证分数（按参数组合索引）")
    plt.xlabel("参数组合索引"); plt.ylabel("平均验证分数"); plt.grid(True); plt.tight_layout()
    savefig("cv_curve.png")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y_test.values) ** 2)))  # 手工 RMSE
    print("\n===== 测试集回归指标（XGB/HistGB + logit）=====")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")
    print(f"RMSE: {rmse:.4f}  (numpy 手工计算)")

    # 预测 vs 实际
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.75)
    plt.xlabel("实际值"); plt.ylabel("预测值"); plt.title("预测 vs 实际（测试集）")
    plt.plot([lo, hi], [lo, hi])
    plt.tight_layout()
    savefig("pred_vs_actual.png")

    # 2) Huber 回归
    huber_pipe = Pipeline(steps=[("preprocess", preprocess), ("model", HuberRegressor())])
    huber_pipe.fit(X_train, y_train)
    y_pred_h = huber_pipe.predict(X_test)
    mae_h = mean_absolute_error(y_test, y_pred_h)
    r2_h  = r2_score(y_test, y_pred_h)
    rmse_h = float(np.sqrt(np.mean((y_pred_h - y_test.values) ** 2)))
    print("\n===== HuberRegressor（鲁棒）指标 =====")
    print(f"MAE:  {mae_h:.4f}, R^2: {r2_h:.4f}, RMSE: {rmse_h:.4f}")

    # 3) Quantile 中位数回归（可能与版本/求解器相关）
    try:
        qreg = QuantileRegressor(quantile=0.5, alpha=1e-4)
        q_pipe = Pipeline(steps=[("preprocess", preprocess), ("model", qreg)])
        q_pipe.fit(X_train, y_train)
        y_pred_q = q_pipe.predict(X_test)
        mae_q = mean_absolute_error(y_test, y_pred_q)
        r2_q  = r2_score(y_test, y_pred_q)
        rmse_q = float(np.sqrt(np.mean((y_pred_q - y_test.values) ** 2)))
        print("\n===== QuantileRegressor（τ=0.5）指标 =====")
        print(f"MAE:  {mae_q:.4f}, R^2: {r2_q:.4f}, RMSE: {rmse_q:.4f}")
    except Exception as e:
        print("\nQuantileRegressor 训练失败：", e)

# ========== 特征重要性 ==========
try:
    pre = best_model.named_steps["preprocess"]
    feature_names = list(pre.get_feature_names_out())
except Exception:
    # 兜底：根据变换后维度造名
    sample_trans = preprocess.fit_transform(X_all.iloc[:1])
    n_feats = sample_trans.shape[1] if hasattr(sample_trans, "shape") else len(sample_trans.ravel())
    feature_names = [f"f{i}" for i in range(n_feats)]

# 模型自带重要性（若不可用则跳过）
try:
    importances = best_model.named_steps["model"].feature_importances_
    k = min(len(feature_names), len(importances))
    feat_imp = pd.DataFrame({"feature": feature_names[:k], "importance": importances[:k]}) \
        .sort_values("importance", ascending=False)
    top_k = min(20, len(feat_imp))
    plt.figure(figsize=(8, max(4, top_k*0.3)))
    plt.barh(range(top_k), feat_imp["importance"].head(top_k)[::-1])
    plt.yticks(range(top_k), feat_imp["feature"].head(top_k)[::-1])
    plt.xlabel("重要性"); plt.title("Top-20 特征重要性"); plt.tight_layout()
    savefig("feature_importance_top20.png")
except Exception:
    feat_imp = pd.DataFrame(columns=["feature", "importance"])

# 置换重要性（更能体现对指标影响；可能较慢）
try:
    if 'y_test_num' in locals():          # 分类
        y_for_perm = np.asarray(y_test_num.values)
    elif 'y_test' in locals():            # 回归
        y_for_perm = np.asarray(y_test.values)
    else:
        y_for_perm = np.asarray(y_raw.values)

    r = permutation_importance(
        best_model, X_test if 'X_test' in locals() else X_all, y_for_perm,
        n_repeats=5, random_state=42, n_jobs=2
    )
    perm_imp = pd.DataFrame({
        "feature": feature_names[:len(r.importances_mean)],
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)

    k2 = min(15, len(perm_imp))
    plt.figure(figsize=(8, max(4, k2*0.3)))
    plt.barh(range(k2), perm_imp["importance_mean"].head(k2)[::-1])
    plt.yticks(range(k2), perm_imp["feature"].head(k2)[::-1])
    plt.xlabel("重要性（对验证分数的平均影响）"); plt.title("Top-15 置换重要性"); plt.tight_layout()
    savefig("permutation_importance_top15.png")
except Exception as e:
    print("置换重要性计算失败：", e)
    perm_imp = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

# ========== 保存产物 ==========
os.makedirs("artifacts", exist_ok=True)
pd.DataFrame(grid.cv_results_).to_csv(os.path.join("artifacts", "cv_results.csv"), index=False)
feat_imp.to_csv(os.path.join("artifacts", "feature_importances.csv"), index=False)
joblib.dump(best_model, os.path.join("artifacts", f"best_model_{'cls' if TASK_MODE=='classification' else 'reg'}.pkl"))

print("\n已保存：")
print(" - 模型：artifacts/best_model_*.pkl")
print(" - 交叉验证结果：artifacts/cv_results.csv")
print(" - 特征重要性：artifacts/feature_importances.csv")
print(" - 图像：cv_curve.png / confusion_matrix.png / roc_curve.png / pred_vs_actual.png / feature_importance_top20.png / permutation_importance_top15.png")
print(f" - 任务类型：{TASK_MODE}")
