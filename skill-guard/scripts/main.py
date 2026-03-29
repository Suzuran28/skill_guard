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

# Windows 终端强制 UTF-8 输出，避免 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 将 src/ 目录加入 Python 搜索路径，避免依赖 chdir
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)

try:
    from src import check_file
    from src.security_engine import SecurityEngine
except ImportError as _e:
    print(f"[错误] 缺少依赖模块：{_e}")
    print("请先运行: python setup_env.py")
    sys.exit(1)

# 纳入扫描的文件扩展名（与训练数据保持一致）
_TEXT_EXTS = {".md", ".py", ".sh", ".js", ".ts", ".rb", ".go", ".ps1", ".bat", ".cmd", ".txt"}
_KEY_DIRS = ("scripts", "references", "assets")


def filter_skill_files(skill_root: str, file_paths: list[str]) -> list[str]:
    """从 check_single_skill 返回的路径列表中筛选关键文件。

    保留规则（优先级从高到低）：
    1. SKILL.md
    2. scripts/ references/ assets/ 子目录下的文本文件
    3. skill 根目录直接子文本文件

    Args:
        skill_root (str): skill 根目录绝对路径。
        file_paths (list[str]): check_single_skill 返回的所有文件路径。

    Returns:
        list[str]: 筛选后的文件路径列表，保持原顺序。
    """
    skill_root_norm = os.path.normpath(skill_root)
    result: list[str] = []
    for p in file_paths:
        p_norm = os.path.normpath(p)
        fname = os.path.basename(p_norm)
        ext = os.path.splitext(fname)[1].lower()
        # SKILL.md 无条件纳入
        if fname == "SKILL.md":
            result.append(p)
            continue
        if ext not in _TEXT_EXTS:
            continue
        # 判断所在目录：关键子目录 或 skill 根目录直属文件
        rel = os.path.relpath(p_norm, skill_root_norm)
        parts = rel.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[0] in _KEY_DIRS:
            result.append(p)
        elif len(parts) == 1:
            result.append(p)
    return result


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
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[错误] 缺少依赖：{e}")
        print("请先运行: python setup_env.py")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        print("未找到模型文件，请重新下载skill")
        sys.exit(1)

    skills_dir = check_file.get_skills_dir()
    try:
        skills = check_file.check_skill()
    except FileNotFoundError:
        print(f"[错误] skills 目录不存在: {skills_dir}")
        print("请确认 openclaw 已安装，或设置环境变量 SKILL_GUARD_SKILLS_DIR。")
        sys.exit(1)
    if not skills:
        print("未发现需要检查的 skill。")
        return

    print(f"\n发现 {len(skills)} 个待检查 skill\n")
    print(f"{'Skill':<40} {'风险等级':<12} {'置信度'}")
    print("-" * 65)

    _LEVEL_ORDER = {"safe": 0, "low": 1, "medium": 2, "high": 3}

    high_risk: list[str] = []

    for skill in skills:
        skill_root = os.path.join(skills_dir, skill)
        all_files = check_file.check_single_skill(skill, skills_dir)
        key_files = filter_skill_files(skill_root, all_files)
        if not key_files:
            print(f"{skill:<40} {'':12}  (无可读文件，跳过)")
            continue

        # 逐文件预测，收集命中原因与对应文件
        final_label = "safe"
        final_confidence = 1.0
        file_hits: list[tuple[str, list[str]]] = []  # (rel_path, reasons)

        for fp in key_files:
            result = engine.predict_file(fp)
            if result is None:
                continue
            rel = os.path.relpath(fp, skill_root).replace("\\", "/")
            if result.reasons:
                file_hits.append((rel, result.reasons))
            if _LEVEL_ORDER[result.label] > _LEVEL_ORDER[final_label]:
                final_label = result.label
                final_confidence = result.confidence
            elif _LEVEL_ORDER[result.label] == _LEVEL_ORDER[final_label]:
                # 同级取置信度更高的
                if result.confidence > final_confidence:
                    final_confidence = result.confidence

        label_str = label_colored(final_label)
        print(f"{skill:<40} {label_str}  {final_confidence:.1%}")
        for rel, reasons in file_hits:
            for reason in reasons:
                print(f"  {'':40}   ↳ {rel} -> {reason}")

        if final_label == "high":
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
