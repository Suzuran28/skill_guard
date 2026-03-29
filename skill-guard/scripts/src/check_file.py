"""Skill 文件扫描工具。

提供 skill 目录发现、白名单过滤与文件枚举功能。
skills 目录和白名单路径支持三种来源（优先级从高到低）：
  1. 函数参数显式传入
  2. 环境变量 SKILL_GUARD_SKILLS_DIR / SKILL_GUARD_WHITELIST
  3. 默认路径 ~/.openclaw/workspace/skills
"""

from __future__ import annotations

import os

DEFAULT_PATH = r"~/.openclaw/workspace/skills"
# DEFAULT_PATH = r"./"    # 测试环境

# ─────────────────────── 路径解析 ───────────────────────

def get_skills_dir() -> str:
    """获取 skills 根目录路径。

    Returns:
        str: 优先取环境变量 SKILL_GUARD_SKILLS_DIR，否则返回默认路径。
    """
    return os.environ.get("SKILL_GUARD_SKILLS_DIR") or os.path.expanduser(DEFAULT_PATH)


def get_whitelist_path(skills_dir: str | None = None) -> str:
    """获取白名单文件路径。

    Args:
        skills_dir (str | None): skills 根目录，为 None 时调用 get_skills_dir()。

    Returns:
        str: 优先取环境变量 SKILL_GUARD_WHITELIST，否则返回
            <skills_dir>/skill-guard/whitelist.txt。
    """
    env = os.environ.get("SKILL_GUARD_WHITELIST")
    if env:
        return env
    base = skills_dir or get_skills_dir()
    return os.path.join(base, "skill-guard", "whitelist.txt")


# ─────────────────────── 核心函数 ───────────────────────

def check_skill(
    skills_dir: str | None = None,
    whitelist_path: str | None = None,
) -> list[str]:
    """查找已下载且不在白名单中的 skill。

    Args:
        skills_dir (str | None): skills 根目录；为 None 时自动解析。
        whitelist_path (str | None): 白名单文件路径；为 None 时自动解析。

    Returns:
        list[str]: 不在白名单中的 skill 名称列表。

    Raises:
        FileNotFoundError: skills 目录不存在时抛出。
    """
    base = skills_dir or get_skills_dir()
    wl_path = whitelist_path or get_whitelist_path(base)

    all_entries = os.listdir(base)

    whitelist_names: set[str] = set()
    if os.path.isfile(wl_path):
        with open(wl_path, "r", encoding="utf-8") as f:
            whitelist_names = {line.strip() for line in f if line.strip()}

    return [e for e in all_entries if e not in whitelist_names]


def check_filefolder(prefix: str) -> list[str]:
    """递归枚举目录下的所有文件。

    Args:
        prefix (str): 要枚举的根目录绝对路径。

    Returns:
        list[str]: 目录下所有文件的绝对路径列表。
    """
    result: list[str] = []
    for entry in os.listdir(prefix):
        full = os.path.join(prefix, entry)
        if os.path.isdir(full):
            result.extend(check_filefolder(full))
        else:
            result.append(full)
    return result


def check_single_skill(
    skill_name: str,
    skills_dir: str | None = None,
) -> list[str]:
    """枚举单个 skill 目录下的所有文件。

    Args:
        skill_name (str): skill 名称（skills 根目录下的子目录名）。
        skills_dir (str | None): skills 根目录；为 None 时自动解析。

    Returns:
        list[str]: skill 目录下所有文件的绝对路径列表。
    """
    base = skills_dir or get_skills_dir()
    return check_filefolder(os.path.join(base, skill_name))


if __name__ == "__main__":
    import sys

    _dir = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"skills 目录: {_dir or get_skills_dir()}")
    for _skill in check_skill(_dir):
        print(f"Skill: {_skill}")
        for _file in check_single_skill(_skill, _dir):
            print(f"  - {_file}")
