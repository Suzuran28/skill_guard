"""Skill Guard 主入口。

检查当前 skills 目录下不在白名单中的 skill，
使用 TF-IDF + XGBoost 安全引擎对每个 skill 的文件进行风险评估，
输出风险等级（safe / low / medium / high）。

用法:
    python main.py

输出:
    扫描结果打印到标准输出，high 风险 skill 会有明显警告。
"""

from __future__ import annotations

import sys
import os

# check_file.py 的路径基于项目根目录，切换到正确的工作目录
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

from src import check_file
from src.security_engine import SecurityEngine

# 风险等级颜色（ANSI）
_COLORS = {
    "safe":   "\033[32m",  # 绿
    "low":    "\033[33m",  # 黄
    "medium": "\033[35m",  # 紫
    "high":   "\033[31m",  # 红
}
_RESET = "\033[0m"


def label_colored(label: str) -> str:
    color = _COLORS.get(label, "")
    return f"{color}{label.upper():8}{_RESET}"


def main() -> None:
    print("正在加载安全引擎...")
    try:
        engine = SecurityEngine()
    except FileNotFoundError as e:
        print(f"错误：{e}")
        print("请先运行 src_model/02_train_model.py 训练并保存模型。")
        sys.exit(1)

    skills = check_file.check_skill()
    if not skills:
        print("未发现需要检查的 skill。")
        return

    print(f"\n发现 {len(skills)} 个待检查 skill\n")
    print(f"{'Skill':<40} {'风险等级':<12} {'置信度'}")
    print("-" * 65)

    high_risk: list[str] = []

    for skill in skills:
        file_paths = check_file.check_single_skill(skill)
        # 优先扫描 SKILL.md，无则扫描全部文件拼接
        skill_md = [p for p in file_paths if os.path.basename(p).upper() == "SKILL.MD"]
        targets = skill_md if skill_md else file_paths

        # 合并文本（SKILL.md 通常只有一个）
        texts = []
        for path in targets:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
            except OSError:
                pass
        combined = "\n".join(texts).strip()
        if not combined:
            continue

        result = engine.predict(combined)
        label_str = label_colored(result.label)
        print(f"{skill:<40} {label_str}  {result.confidence:.1%}")

        if result.label == "high":
            high_risk.append(skill)

    print("-" * 65)

    if high_risk:
        print(f"\n\033[31m警告：发现 {len(high_risk)} 个高风险 skill！\033[0m")
        for s in high_risk:
            print(f"  \033[31m[HIGH] {s}\033[0m")
        sys.exit(2)  # 非零退出码方便自动化流水线检测
    else:
        print("\n\033[32m所有 skill 已通过安全检查。\033[0m")


if __name__ == "__main__":
    main()
