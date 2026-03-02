import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, RocCurveDisplay

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

for c_test in [0.005, 0.01, 0.05, 0.1]:
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

# --- 步骤 3: 数据准备与模型训练 ---
X_final = df_final.drop(columns=['hospital_expire_flag'])
y_final = df_final['hospital_expire_flag']

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练三个模型
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)

# --- 步骤 4: 结果打印与全指标评估 (全功能加强版) ---
from sklearn.metrics import recall_score, precision_score

def evaluate_model_full(model, X_tr, y_tr, X_te, y_te, name, is_scaled=False):
    # 自动选择对应的特征集
    xtr = X_tr
    xte = X_te
    
    # 1. 计算各种指标 (测试集)
    probs = model.predict_proba(xte)[:, 1]
    preds = model.predict(xte)
    
    auc_te = roc_auc_score(y_te, probs)
    acc_te = accuracy_score(y_te, preds)
    f1_te = f1_score(y_te, preds)
    pre_te = precision_score(y_te, preds)
    rec_te = recall_score(y_te, preds)
    
    # 2. 计算训练集 AUC 用于检测过拟合
    auc_tr = roc_auc_score(y_tr, model.predict_proba(xtr)[:, 1])
    gap = auc_tr - auc_te
    
    # 3. 打印结果
    print(f"[{name}]")
    print(f"  - 测试集: AUC={auc_te:.4f} | F1={f1_te:.4f} | Acc={acc_te:.4f} | Precision={pre_te:.4f} | Recall={rec_te:.4f}")
    print(f"  - 过拟合检测: Train AUC={auc_tr:.4f} (Gap: {gap:.4f})")
    print("-" * 70)
    
    # 返回结果用于后续对比（可选）
    return {"Model": name, "AUC": auc_te, "F1": f1_te, "Acc": acc_te, "Gap": gap}

print(f"{'模型评估报告':^60}")

results = []
results.append(evaluate_model_full(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression", True))
results.append(evaluate_model_full(dt_model, X_train, y_train, X_test, y_test, "Decision Tree"))
results.append(evaluate_model_full(rf_model, X_train, y_train, X_test, y_test, "Random Forest"))

# 4. 生成一个可以直接贴进报告的汇总 DataFrame
summary_df = pd.DataFrame(results)
print("\n>>> 最终对比摘要 (可以直接用于报告表格):")
print(summary_df.to_string(index=False))
print("="*70)

# --- 步骤 5: 绘图 ---
# ROC 曲线
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(lr_model, X_test_scaled, y_test, ax=ax, name='LR')
RocCurveDisplay.from_estimator(dt_model, X_test, y_test, ax=ax, name='DT')
RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax, name='RF')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("ROC Comparison (37 Selected Features)")
plt.savefig('roc_comparison.png')

# 核心特征相关性热力图
plt.figure(figsize=(12, 10))
# 获取 RF 重要性排名前 15 的特征用于画图
top_15_names = pd.Series(rf_model.feature_importances_, index=selected_feature_names).sort_values(ascending=False).head(15).index
corr_subset = X_final[top_15_names].corr()
mask = np.triu(np.ones_like(corr_subset, dtype=bool))
sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap='RdBu_r', center=0, mask=mask)
plt.title('Top 15 Features Correlation')
plt.savefig('2_correlation_heatmap.png')

# 特征重要性条形图
plt.figure(figsize=(10, 8))
feat_importances = pd.Series(rf_model.feature_importances_, index=selected_feature_names).sort_values(ascending=False).head(20)
sns.barplot(x=feat_importances.values, y=feat_importances.index, palette='magma')
plt.title('Top 20 Features by RF Importance')
plt.savefig('3_feature_importance.png')

print("\n所有分析完成！图片已保存。")