"""使用训练好的模型扫描 skill 文件，输出风险等级。

加载 TF-IDF + XGBoost 模型，对指定目录下所有 SKILL.md 文件进行风险分类，
输出每个 skill 的风险等级和置信度。

用法:
    python 03_scan.py [skill_dir]

    skill_dir: 要扫描的 skill 根目录，默认为当前目录下的 test_skills/

依赖:
    pip install scikit-learn xgboost

输入:
    model/tfidf_vectorizer.pkl
    model/xgboost_classifier.pkl
    model/label_encoder.pkl
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np


RISK_COLORS = {
    "safe": "\033[32m",      # 绿色
    "low": "\033[33m",       # 黄色
    "medium": "\033[35m",    # 紫色
    "high": "\033[31m",      # 红色
}
RESET = "\033[0m"


def load_pickle(path: str) -> object:
    """从 pickle 文件加载对象。

    Args:
        path (str): pickle 文件路径。

    Returns:
        object: 反序列化后的对象。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def find_skill_files(root: str) -> list[str]:
    """递归查找目录下所有 SKILL.md 文件。

    Args:
        root (str): 搜索根目录。

    Returns:
        list[str]: 所有 SKILL.md 文件的绝对路径列表。
    """
    results = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname == "SKILL.md":
                results.append(os.path.join(dirpath, fname))
    return results


def predict_risk(text: str, vectorizer, clf, le) -> tuple[str, float]:
    """对单条文本预测风险等级。

    Args:
        text (str): skill 文本内容。
        vectorizer: 已训练的 TF-IDF 向量化器。
        clf: 已训练的 XGBoost 分类器。
        le: 已训练的标签编码器。

    Returns:
        tuple[str, float]: (风险等级标签, 该等级的置信度 0~1)。
    """
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label = le.inverse_transform([idx])[0]
    confidence = float(proba[idx])
    return label, confidence


def main() -> None:
    skill_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    model_dir = os.path.join(os.path.dirname(__file__), "model")
    if not os.path.isdir(model_dir):
        model_dir = "model"

    try:
        vectorizer = load_pickle(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        clf = load_pickle(os.path.join(model_dir, "xgboost_classifier.pkl"))
        le = load_pickle(os.path.join(model_dir, "label_encoder.pkl"))
    except FileNotFoundError as e:
        print(f"错误: 模型文件未找到 ({e})，请先运行 02_train_model.py")
        sys.exit(1)

    skill_files = find_skill_files(skill_dir)
    if not skill_files:
        print(f"未在 {skill_dir} 中找到任何 SKILL.md 文件")
        return

    print(f"扫描目录: {skill_dir}，共找到 {len(skill_files)} 个 skill\n")
    print(f"{'Skill 路径':<60} {'风险等级':<10} {'置信度'}")
    print("-" * 80)

    summary: dict[str, list[str]] = {"safe": [], "low": [], "medium": [], "high": []}

    for path in sorted(skill_files):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        label, confidence = predict_risk(text, vectorizer, clf, le)
        color = RISK_COLORS.get(label, "")
        summary[label].append(path)

        rel_path = os.path.relpath(path, skill_dir)
        print(f"{rel_path:<60} {color}{label:<10}{RESET} {confidence:.1%}")

    print("-" * 80)
    print("\n扫描摘要:")
    for level in ["high", "medium", "low", "safe"]:
        count = len(summary[level])
        if count:
            color = RISK_COLORS[level]
            print(f"  {color}{level.upper():<8}{RESET}: {count} 个")


if __name__ == "__main__":
    main()
