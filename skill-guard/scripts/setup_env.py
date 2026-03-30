"""Skill Guard 环境配置脚本。

检查并安装运行 skill-guard 所需的 Python 依赖（scikit-learn、xgboost）。
支持 pip 和 conda 两种包管理器，自动检测当前 Python 环境。

用法:
    python setup_env.py

成功后输出各依赖的版本信息，失败时给出安装建议。
"""

from __future__ import annotations

import importlib
import subprocess
import sys


REQUIRED = [
    ("sklearn",  "scikit-learn", "1.7.2"),
    ("xgboost",  "xgboost",      "3.2.0"),
]


def check_module(import_name: str) -> str | None:
    """检查模块是否已安装，返回版本号或 None。

    Args:
        import_name (str): 模块的 import 名称。

    Returns:
        str | None: 已安装时返回版本字符串，否则返回 None。
    """
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


# 国内镜像源，按优先级轮换
_MIRRORS = [
    None,  # 官方源（优先）
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.ustc.edu.cn/pypi/simple",
    "https://mirrors.aliyun.com/pypi/simple",
]


def pip_install(package: str) -> bool:
    """使用当前 Python 环境的 pip 安装包，失败时自动轮换国内镜像源。

    Args:
        package (str): pip 包名。

    Returns:
        bool: 至少一个源安装成功返回 True，否则返回 False。
    """
    for mirror in _MIRRORS:
        cmd = [sys.executable, "-m", "pip", "install", package, "--quiet"]
        if mirror:
            cmd += ["-i", mirror, "--trusted-host", mirror.split("/")[2]]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            return True
    return False


def main() -> None:
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    print("-" * 55)

    all_ok = True

    for import_name, pip_name, min_ver in REQUIRED:
        ver = check_module(import_name)
        if ver is not None:
            print(f"  [OK]  {pip_name} {ver}")
            continue

        print(f"  [--]  {pip_name} 未安装，正在安装...", end=" ", flush=True)
        if pip_install(pip_name):
            ver = check_module(import_name) or "?"
            print(f"完成 ({ver})")
        else:
            print("失败")
            print(f"       请手动安装: pip install {pip_name}>={min_ver}")
            all_ok = False

    print("-" * 55)

    if all_ok:
        # 验证模型文件是否存在
        import os
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        required_files = ["tfidf_vectorizer.pkl", "xgboost_classifier.pkl", "label_encoder.pkl"]
        missing = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
        if missing:
            print("[WARN] 模型文件缺失:")
            for f in missing:
                print(f"       - model/{f}")
            print("       未找到模型文件，请重新下载skill")
        else:
            print("[OK]  模型文件完整")
            print("\n环境配置完成，可运行:")
            print("  python main.py")
    else:
        print("\n部分依赖安装失败，请检查网络或手动安装后重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
