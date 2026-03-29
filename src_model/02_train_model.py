"""训练 skill 安全风险分类模型。

使用 TF-IDF 提取文本特征，XGBoost 训练四分类模型（safe/low/medium/high）。
训练完成后保存模型文件和标签映射。

用法:
    python 02_train_model.py

依赖:
    pip install scikit-learn xgboost

输入:
    data/train.jsonl  训练集
    data/val.jsonl    验证集

输出:
    model/tfidf_vectorizer.pkl  TF-IDF 向量化器
    model/xgboost_classifier.pkl  XGBoost 分类器
    model/label_encoder.pkl  标签编码器
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def load_jsonl(path: str) -> tuple[list[str], list[str]]:
    """从 JSONL 文件加载文本和标签。

    Args:
        path (str): JSONL 文件路径，每行含 text 和 label 字段。

    Returns:
        tuple[list[str], list[str]]: (文本列表, 标签列表)。
    """
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])
    return texts, labels


def save_pickle(obj: object, path: str) -> None:
    """将对象序列化保存为 pickle 文件。

    Args:
        obj (object): 待保存的对象。
        path (str): 输出文件路径。
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main() -> None:
    os.makedirs("model", exist_ok=True)

    print("加载数据...")
    train_texts, train_labels = load_jsonl("data/train.jsonl")
    val_texts, val_labels = load_jsonl("data/val.jsonl")
    print(f"  训练集: {len(train_texts)} 条，验证集: {len(val_texts)} 条")

    # 标签编码
    le = LabelEncoder()
    le.fit(train_labels + val_labels)
    y_train = le.transform(train_labels)
    y_val = le.transform(val_labels)
    print(f"  标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # TF-IDF 特征提取
    print("提取 TF-IDF 特征...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    print(f"  特征维度: {X_train.shape[1]}")

    # XGBoost 训练
    print("训练 XGBoost 模型...")
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    # 计算类权重以处理样本不均衡
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    clf.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 评估
    y_pred = clf.predict(X_val)
    print("\n验证集评估结果:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # 保存模型
    save_pickle(vectorizer, "model/tfidf_vectorizer.pkl")
    save_pickle(clf, "model/xgboost_classifier.pkl")
    save_pickle(le, "model/label_encoder.pkl")
    print("模型已保存至 model/ 目录")


if __name__ == "__main__":
    main()
