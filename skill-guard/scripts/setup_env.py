"""Skill Guard 环境配置脚本。

检查并安装运行 skill-guard 所需的 Python 依赖（scikit-learn、xgboost）。
启动时并发探测所有镜像源响应时间，选最快的一个用于安装。

用法:
    python setup_env.py

成功后输出各依赖的版本信息，失败时给出安装建议。
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


REQUIRED = [
    ("sklearn", "scikit-learn", "1.8.0"),
    ("xgboost", "xgboost",      "3.2.0"),
]

# 候选镜像源；None 代表官方 pypi.org
_MIRRORS: list[str | None] = [
    None,
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.ustc.edu.cn/pypi/simple",
    "https://mirrors.aliyun.com/pypi/simple",
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


def _probe(mirror: str | None) -> tuple[float, str | None]:
    """对单个镜像源发起 HTTP 请求，返回 (耗时秒, mirror)。超时返回 inf。

    Args:
        mirror (str | None): 镜像源 URL，None 代表官方源。

    Returns:
        tuple[float, str | None]: (响应耗时, mirror)。
    """
    url = (mirror or "https://pypi.org/simple") + "/pip/"
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(url, timeout=5):
            pass
        return time.monotonic() - t0, mirror
    except Exception:
        return float("inf"), mirror


def pick_fastest_mirror() -> str | None:
    """并发探测所有候选镜像源，返回响应最快的一个。

    Returns:
        str | None: 最快镜像源的 URL；None 表示官方源最快或全部超时。
    """
    print("正在探测镜像源速度...", flush=True)
    results: list[tuple[float, str | None]] = []

    with ThreadPoolExecutor(max_workers=len(_MIRRORS)) as pool:
        futures = {pool.submit(_probe, m): m for m in _MIRRORS}
        for fut in as_completed(futures):
            elapsed, mirror = fut.result()
            label = mirror or "pypi.org (官方)"
            if elapsed == float("inf"):
                print(f"  [超时] {label}")
            else:
                print(f"  [{elapsed * 1000:.0f}ms] {label}")
            results.append((elapsed, mirror))

    results.sort(key=lambda x: x[0])
    best_elapsed, best_mirror = results[0]

    if best_elapsed == float("inf"):
        print("  → 所有源均超时，使用官方源兜底")
        return None

    print(f"  → 最快: {best_mirror or 'pypi.org (官方)'}")
    return best_mirror


def pip_install(package: str, mirror: str | None) -> bool:
    """使用指定镜像源安装 pip 包。

    Args:
        package (str): pip 包名。
        mirror (str | None): 镜像源 URL；None 使用官方源。

    Returns:
        bool: 安装成功返回 True，否则返回 False。
    """
    cmd = [sys.executable, "-m", "pip", "install", package, "--quiet"]
    if mirror:
        cmd += ["-i", mirror, "--trusted-host", mirror.split("/")[2]]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main() -> None:
    import os

    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    print("-" * 55)

    # 探测最快镜像源（仅在有包需要安装时才有意义，但先探测避免安装时再等待）
    need_install = any(check_module(imp) is None for imp, _, _ in REQUIRED)
    mirror = pick_fastest_mirror() if need_install else None
    if need_install:
        print("-" * 55)

    all_ok = True

    for import_name, pip_name, min_ver in REQUIRED:
        ver = check_module(import_name)
        if ver is not None:
            print(f"  [OK]  {pip_name} {ver}")
            continue

        print(f"  [--]  {pip_name} 未安装，正在安装...", end=" ", flush=True)
        if pip_install(pip_name, mirror):
            ver = check_module(import_name) or "?"
            print(f"完成 ({ver})")
        else:
            print("失败")
            print(f"       请手动安装: pip install {pip_name}>={min_ver}")
            all_ok = False

    print("-" * 55)

    if all_ok:
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        required_files = ["tfidf_vectorizer.pkl", "xgboost_classifier.pkl", "label_encoder.pkl"]
        missing = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
        if missing:
            print("[WARN] 模型文件缺失:")
            for f in missing:
                print(f"       - model/{f}")
            print("       未找到模型文件，请重新下载 skill")
        else:
            print("[OK]  模型文件完整")
            print("\n环境配置完成，可运行:")
            print("  python main.py")
    else:
        print("\n部分依赖安装失败，请检查网络或手动安装后重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
