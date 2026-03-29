"""Skill 安全风险检测引擎。

加载预训练的 TF-IDF + XGBoost 模型，对 skill 文本进行风险等级预测。
风险等级分为四级：safe / low / medium / high。

用法示例:
    engine = SecurityEngine()
    result = engine.predict(text)
    print(result.label, result.confidence)
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

# 模型文件相对于本文件的路径
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


@dataclass
class PredictResult:
    """单条文本的预测结果。

    Attributes:
        label (str): 风险等级，safe / low / medium / high 之一。
        confidence (float): 预测置信度，0.0 ~ 1.0。
        probabilities (dict[str, float]): 各类别的预测概率。
    """

    label: str
    confidence: float
    probabilities: dict[str, float]


class SecurityEngine:
    """Skill 安全风险检测引擎。

    封装 TF-IDF 向量化器、XGBoost 分类器和标签编码器，提供统一的预测接口。
    模型文件从 scripts/model/ 目录加载。
    """

    def __init__(self, model_dir: str = _MODEL_DIR) -> None:
        """初始化安全引擎，加载模型文件。

        Args:
            model_dir (str): 模型文件所在目录，默认为 scripts/model/。

        Raises:
            FileNotFoundError: 当模型文件不存在时抛出。
        """
        self._vectorizer = self._load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        self._classifier = self._load(os.path.join(model_dir, "xgboost_classifier.pkl"))
        self._label_encoder = self._load(os.path.join(model_dir, "label_encoder.pkl"))

    @staticmethod
    def _load(path: str):
        """从 pickle 文件加载对象。

        Args:
            path (str): pickle 文件路径。

        Returns:
            加载的对象。

        Raises:
            FileNotFoundError: 文件不存在时抛出。
        """
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"模型文件不存在: {abs_path}")
        with open(abs_path, "rb") as f:
            return pickle.load(f)

    def predict(self, text: str) -> PredictResult:
        """对单条文本进行风险等级预测。

        Args:
            text (str): 待检测的 skill 文本内容。

        Returns:
            PredictResult: 包含风险等级、置信度和各类概率的预测结果。
        """
        x = self._vectorizer.transform([text])
        proba = self._classifier.predict_proba(x)[0]
        classes = self._label_encoder.classes_
        idx = int(proba.argmax())
        return PredictResult(
            label=str(classes[idx]),
            confidence=float(proba[idx]),
            probabilities={str(c): float(p) for c, p in zip(classes, proba)},
        )

    def predict_file(self, file_path: str) -> PredictResult | None:
        """读取文件并进行风险等级预测。

        Args:
            file_path (str): 待检测文件路径（通常为 SKILL.md）。

        Returns:
            PredictResult | None: 预测结果，文件读取失败时返回 None。
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except OSError:
            return None
        if not text:
            return None
        return self.predict(text)
