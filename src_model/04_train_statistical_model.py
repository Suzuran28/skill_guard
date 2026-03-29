import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib

# 1. 加载数据
df = pd.read_csv("skill_stats_v2.csv")
X = df.drop(columns=['label'])
y = df['label']

# 分层拆分 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. 初始化模型
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    tree_method='hist',
    device='cuda',
    scale_pos_weight=1.6,
    random_state=42
)

# 3. 训练
print("XGBoost 开始训练...")
model.fit(X_train, y_train)

# 4. 评估
y_pred = model.predict(X_test)
print("\n--- XGBoost 统计模型评估报告 ---")
print(classification_report(y_test, y_pred))

# 5. 特征重要性分析
importances = model.get_booster().get_score(importance_type='weight')
print("\n核心特征贡献排行:")
sorted_idx = sorted(importances.items(), key=lambda x: x[1], reverse=True)
for k, v in sorted_idx[:5]:
    print(f" - {k}: {v}")

# 6. 保存模型
joblib.dump(model, "xgboost_security_v2.pkl")
print("\n统计模型已保存。")