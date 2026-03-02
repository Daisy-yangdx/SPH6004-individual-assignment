import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (recall_score, precision_score, accuracy_score, 
                             f1_score, roc_auc_score, RocCurveDisplay)

# 1. 数据加载与基础清洗
print("正在加载数据...")
df = pd.read_csv('Assignment1_mimic dataset.csv', na_values=['', ' ', 'None', 'NaN', 'nan'])
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

leakage_and_useless = ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'deathtime', 'icu_death_flag', 'los']
df_clean = df.drop(columns=[c for c in leakage_and_useless if c in df.columns])

# 缺失率过滤 (40%)
missing_rate = df_clean.isnull().mean()
df_clean = df_clean.loc[:, missing_rate <= 0.4]

# 异常值剔除
valid_heart = (df_clean['heart_rate_mean'].isna()) | (df_clean['heart_rate_mean'].between(1, 299))
valid_resp  = (df_clean['resp_rate_mean'].isna()) | (df_clean['resp_rate_mean'].between(1, 99))
valid_sbp   = (df_clean['sbp_mean'].isna()) | (df_clean['sbp_mean'].between(1, 299))
df_clean = df_clean[valid_heart & valid_resp & valid_sbp]

# One-Hot 编码
df_clean = pd.get_dummies(df_clean, drop_first=True)
df_clean = df_clean.fillna(df_clean.median())

# 高级特征筛选 (L1 + CV)
scaler = StandardScaler()
X_raw = df_clean.drop(columns=['hospital_expire_flag'])
y = df_clean['hospital_expire_flag']

print(f"\n正在启动快速筛选策略（初始特征: {X_raw.shape[1]} 维）...")

# 1. RF 预过滤
rf_pre = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf_pre.fit(X_raw, y)
active_rf_mask = rf_pre.feature_importances_ > 0.001 
X_rf_filtered = X_raw.loc[:, active_rf_mask]

# 2. 寻找最优 C
best_auc = 0
best_c = 0.1
X_rf_scaled = scaler.fit_transform(X_rf_filtered)

for c_test in [0.005, 0.01, 0.05, 0.1,0.5]:
    lr_test = LogisticRegression(penalty='l1', solver='liblinear', C=c_test, tol=0.01, random_state=42)
    score = cross_val_score(lr_test, X_rf_scaled, y, cv=3, scoring='roc_auc').mean()
    if score > best_auc:
        best_auc = score
        best_c = c_test

# 3. 提取最终特征
final_lr = LogisticRegression(penalty='l1', solver='liblinear', C=best_c, tol=0.01, random_state=42)
final_lr.fit(X_rf_scaled, y)
active_l1_mask = (final_lr.coef_[0] != 0)
selected_feature_names = X_rf_filtered.columns[active_l1_mask].tolist()

# 更新最终数据集
df_final = df_clean[selected_feature_names + ['hospital_expire_flag']]
print(f"筛选完成！最优 C: {best_c} | 最终核心特征数: {len(selected_feature_names)}")

# --- 步骤 3: 数据准备 (统一标准化) ---
X_final = df_final.drop(columns=['hospital_expire_flag'])
y_final = df_final['hospital_expire_flag']

# 保持 stratify 确保训练集和测试集的死亡比例一致
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 步骤 4: 统一模型评估函数 (提前定义，方便调用) ---
def evaluate_model_full(model, X_tr, y_tr, X_te, y_te, name):
    """
    统一评估函数：自动处理有无 predict_proba 的模型，并计算过拟合 Gap
    """
    if hasattr(model, "predict_proba"):
        probs_te = model.predict_proba(X_te)[:, 1]
        probs_tr = model.predict_proba(X_tr)[:, 1]
    else:
        # 针对 SVM 等没有 predict_proba 的模型使用决策函数
        probs_te = model.decision_function(X_te)
        probs_tr = model.decision_function(X_tr)
    
    preds_te = model.predict(X_te)
    
    auc_te = roc_auc_score(y_te, probs_te)
    auc_tr = roc_auc_score(y_tr, probs_tr)
    acc_te = accuracy_score(y_te, preds_te)
    f1_te = f1_score(y_te, preds_te)
    pre_te = precision_score(y_te, preds_te)
    rec_te = recall_score(y_te, preds_te)
    gap = auc_tr - auc_te 
    
    return {
        "Model": name, 
        "Test AUC": auc_te, 
        "F1-Score": f1_te, 
        "Recall": rec_te, 
        "Precision": pre_te,
        "Accuracy": acc_te,
        "AUC Gap": gap
    }

# --- 步骤 5: 六大算法统一训练与并行评估 ---
print("6算法评估对比")

# 1. Logistic Regression (逻辑回归) - 保持不变
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# 2. SVM (线性支持向量机) - 【修改：加入 class_weight】
# LinearSVC 同样支持 'balanced' 参数
svm_model = LinearSVC(C=0.1, max_iter=10000, class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 3. XGBoost (梯度提升树) - 【修改：加入 scale_pos_weight】
# 计算负样本与正样本的比例（这是 XGBoost 实现 balanced 的方式）
ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = XGBClassifier(
    n_estimators=100, 
    max_depth=4, 
    learning_rate=0.1, 
    scale_pos_weight=ratio,  # 设置正样本权重补偿
    random_state=42, 
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

# 4. AdaBoost (自适应提升) - 【说明：AdaBoost 没有 class_weight 参数】
# 它的原理是根据误差自动调整样本权重。
# 如果想手动加权，必须在 .fit() 时传入 sample_weight。
# 建议保持原样，在报告中写：“AdaBoost 通过迭代误差自动调整权重，不直接使用类权重平衡策略。”
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
# 【核心修改点】：在 fit 时传入 sample_weight
ada_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# 5. Random Forest (随机森林) - 保持不变
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=8, random_state=42)
rf_model.fit(X_train, y_train) 

# 6. Decision Tree (决策树) - 保持不变
dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)

# --- 汇总结果 ---
results = []
results.append(evaluate_model_full(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression"))
results.append(evaluate_model_full(svm_model, X_train_scaled, y_train, X_test_scaled, y_test, "SVM (Linear)"))
results.append(evaluate_model_full(xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, "XGBoost"))
results.append(evaluate_model_full(ada_model, X_train_scaled, y_train, X_test_scaled, y_test, "AdaBoost"))
results.append(evaluate_model_full(rf_model, X_train, y_train, X_test, y_test, "Random Forest"))
results.append(evaluate_model_full(dt_model, X_train, y_train, X_test, y_test, "Decision Tree"))

summary_df = pd.DataFrame(results)
print(" 最终性能汇总表 " )
print(summary_df.to_string(index=False))

# --- 步骤 6: 学术图表生成 ---

# 图表 1: 六模型综合 ROC 曲线对比
plt.figure(figsize=(10, 8))
ax = plt.gca()
plot_configs = [
    (lr_model, X_test_scaled, 'LR (AUC: {:.4f})'),
    (svm_model, X_test_scaled, 'SVM (AUC: {:.4f})'),
    (xgb_model, X_test_scaled, 'XGBoost (AUC: {:.4f})'),
    (ada_model, X_test_scaled, 'AdaBoost (AUC: {:.4f})'),
    (rf_model, X_test, 'RF (AUC: {:.4f})'),
    (dt_model, X_test, 'DT (AUC: {:.4f})')
]

for model, xt, label_fmt in plot_configs:
    if hasattr(model, "predict_proba"):
        cur_auc = roc_auc_score(y_test, model.predict_proba(xt)[:, 1])
    else:
        cur_auc = roc_auc_score(y_test, model.decision_function(xt))
    RocCurveDisplay.from_estimator(model, xt, y_test, ax=ax, name=label_fmt.format(cur_auc))

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Random Guess')
plt.title("ROC Comparison of Six Models (37 Features)", fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('1_six_models_roc.png', dpi=300, bbox_inches='tight')
plt.show()

# 图表 2: 核心特征相关性热力图 (保持 Top 15，最具可读性)
plt.figure(figsize=(12, 10))
top_15_features = pd.Series(rf_model.feature_importances_, index=selected_feature_names).sort_values(ascending=False).head(15).index
corr_matrix = X_final[top_15_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0, mask=mask, square=True)
plt.title('Correlation Heatmap of Top 15 Clinical Predictors', fontsize=14)
plt.savefig('2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 图表 3: 前 20 个特征重要性排序
plt.figure(figsize=(10, 8)) # 高度从 12 减为 8，比例更协调
top_20_feat_importances = pd.Series(rf_model.feature_importances_, index=selected_feature_names).sort_values(ascending=False).head(20)
sns.barplot(x=top_20_feat_importances.values, y=top_20_feat_importances.index, palette='magma') # 换成 magma 色系更显眼
plt.title('Top 20 Clinical Predictors by Feature Importance', fontsize=14)
plt.xlabel('Gini Importance', fontsize=12)
plt.ylabel('Clinical Features', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('3_top20_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n"  + " 所有分析任务已完成！ " )