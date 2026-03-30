"""Skill 安全风险检测引擎。

加载预训练的 TF-IDF + XGBoost 模型，结合规则引擎对 skill 文本进行风险等级预测。
风险等级分为四级：safe / low / medium / high。
规则命中结果与 ML 预测取更高风险等级，规则命中时附带命中原因。

用法示例:
    engine = SecurityEngine()
    result = engine.predict(text)
    print(result.label, result.confidence, result.reasons)
"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field

# 模型文件相对于本文件的路径
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

# 风险等级数值（用于比较高低）
_LEVEL_ORDER = {"safe": 0, "low": 1, "medium": 2, "high": 3}


# ─────────────────────────── 规则定义 ───────────────────────────
# 每条规则：(pattern, 风险等级, 描述)
# 使用 re.IGNORECASE | re.MULTILINE
_RULES: list[tuple[re.Pattern[str], str, str]] = []


def _r(pattern: str, level: str, desc: str) -> None:
    _RULES.append((re.compile(pattern, re.IGNORECASE | re.MULTILINE), level, desc))


# high
_r(r"disable.{0,20}(antivirus|defender|firewall|security)",      "high", "尝试禁用安全软件")
_r(r"(password|passwd).{0,10}(zip|archive|compress)",            "high", "下载密码压缩包")
_r(r"\b(rm\s+-rf|del\s+/[sqf]|format\s+[a-z]:)",               "high", "危险删除/格式化命令")
_r(r"(exfiltrat|steal|dump).{0,20}(password|credential|token|secret)", "high", "数据窃取行为")
_r(r"(reverse.?shell|bind.?shell|netcat.{0,10}-e)",              "high", "反弹/绑定 shell")
_r(r"(keylog|screen.?capture|screenshot).{0,20}(send|upload|post)", "high", "键盘记录/屏幕截图并外发")
_r(r"(curl|wget|invoke-webrequest).{0,60}\|\s*(bash|sh|python|powershell)", "high", "下载并执行远程脚本")
_r(r"(chmod|icacls).{0,20}(777|\+x).{0,40}\.exe",               "high", "赋予可执行权限")
_r(r"base64.{0,30}(decode|\-d).{0,60}(exec|eval|bash|sh|python|invoke)",  "high", "Base64 解码后执行")
_r(r"powershell.{0,20}-encodedcommand",                              "high", "PowerShell EncodedCommand 执行")
_r(r"(curl|wget).{0,80}\.(exe|sh|ps1|bat|cmd).{0,40}(&&|;|\|).{0,20}(bash|sh|\./)?",  "high", "下载到本地再执行")
_r(r"(unzip|tar|expand-archive).{0,80}\.(exe|sh|ps1|bat)",           "high", "解压后运行可执行文件")
_r(r"(\.bashrc|\.zshrc|\.profile|\.bash_profile).{0,40}(echo|append|write|>>)", "high", "修改 shell profile 自启动")
_r(r"authorized_keys",                                                "high", "修改 SSH authorized_keys")
_r(r"(cookie|token|session).{0,30}(sqlite|db|chrome|firefox|edge|safari)", "high", "读取浏览器 cookie/token")

# medium
_r(r"\b(subprocess|os\.system|os\.popen|eval|exec)\s*\(",        "medium", "动态代码/系统命令执行")
_r(r"(requests\.(get|post)|urllib|httpx|aiohttp).{0,80}(upload|exfil|send)", "medium", "网络数据外发")
_r(r"(sudo|runas|privilege.?escal|uac.?bypass)",                 "medium", "提权操作")
_r(r"(registry|regedit|reg\s+add|reg\s+delete)",                 "medium", "注册表操作")
_r(r"(crontab|at\s+command|schtasks|launchd).{0,40}(install|add|creat)", "medium", "持久化计划任务")
_r(r"(socket|listen|bind).{0,30}\d{4,5}",                        "medium", "网络监听")

# low
_r(r"(open|read|write|os\.path|pathlib|glob).{0,30}(password|secret|credential|token)", "low", "访问敏感文件路径")
_r(r"(shutil|zipfile|tarfile).{0,20}(extract|compress)",         "low", "文件压缩/解压")
_r(r"(os\.environ|getenv).{0,30}(password|secret|key|token)",    "low", "读取敏感环境变量")


# ─────────────────────────── 数据类 ───────────────────────────
@dataclass
class PredictResult:
    """单条文本的预测结果。

    Attributes:
        label (str): 风险等级，safe / low / medium / high 之一。
        confidence (float): 预测置信度，0.0 ~ 1.0。
        probabilities (dict[str, float]): 各类别的预测概率（来自 ML 模型）。
        reasons (list[str]): 规则命中原因列表；未命中规则时为空。
    """

    label: str
    confidence: float
    probabilities: dict[str, float]
    reasons: list[str] = field(default_factory=list)


# ─────────────────────────── 引擎 ───────────────────────────
class SecurityEngine:
    """Skill 安全风险检测引擎。

    封装 TF-IDF 向量化器、XGBoost 分类器、标签编码器以及规则引擎，
    提供统一的混合预测接口（规则优先，ML 兜底）。
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

    @staticmethod
    def _match_rules(text: str) -> tuple[str, float, list[str]]:
        """对文本运行所有规则，返回最高风险等级、置信度和命中原因列表。

        Args:
            text (str): 待检测文本。

        Returns:
            tuple[str, float, list[str]]:
                - 命中的最高风险等级（无命中时为 "safe"）
                - 规则置信度（固定 1.0，未命中时 0.0）
                - 命中原因列表
        """
        best_level = "safe"
        reasons: list[str] = []
        seen: set[str] = set()

        for pattern, level, desc in _RULES:
            if pattern.search(text):
                if desc not in seen:
                    seen.add(desc)
                    reasons.append(desc)
                if _LEVEL_ORDER[level] > _LEVEL_ORDER[best_level]:
                    best_level = level

        confidence = 1.0 if reasons else 0.0
        return best_level, confidence, reasons

    def predict(self, text: str) -> PredictResult:
        """对单条文本进行混合风险等级预测（规则引擎 + ML 模型）。

        规则引擎与 ML 模型同时运行，取更高风险等级作为最终结果。
        规则命中时优先使用规则置信度，否则使用 ML 置信度。

        Args:
            text (str): 待检测的 skill 文本内容。

        Returns:
            PredictResult: 包含风险等级、置信度、各类概率和命中原因的预测结果。
        """
        # ML 预测
        x = self._vectorizer.transform([text])
        proba = self._classifier.predict_proba(x)[0]
        classes = self._label_encoder.classes_
        ml_idx = int(proba.argmax())
        ml_label = str(classes[ml_idx])
        ml_confidence = float(proba[ml_idx])
        probabilities = {str(c): float(p) for c, p in zip(classes, proba)}

        # 规则预测
        rule_label, rule_confidence, reasons = self._match_rules(text)

        # 取更高风险等级
        if _LEVEL_ORDER[rule_label] >= _LEVEL_ORDER[ml_label]:
            label = rule_label
            confidence = rule_confidence if reasons else ml_confidence
        else:
            label = ml_label
            confidence = ml_confidence

        return PredictResult(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            reasons=reasons,
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
