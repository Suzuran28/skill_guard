"""生成 skill 安全风险分类的训练数据。

从 Skills/Dangerous_skills 和 Skills/Safe_skills 目录提取真实样本，
并补充手工标注的 low/medium 风险样本，输出为 JSONL 格式。

用法:
    python 01_generate_data.py [--skills-dir PATH]

参数:
    --skills-dir  Skills 数据集根目录，默认为 ../Skills

输出:
    data/train.jsonl  训练集（80%）
    data/val.jsonl    验证集（20%）
"""

from __future__ import annotations

import argparse
import json
import os
import random

random.seed(42)

# ---------------------------------------------------------------------------
# 手工标注的 low / medium 样本（真实数据中缺少中间类）
# ---------------------------------------------------------------------------

LOW_SAMPLES = [
    ("git-helper", "Runs git status, log, and diff commands to help manage your repository."),
    ("file-organizer", "Moves and renames files in your workspace based on instructions."),
    ("log-analyzer", "Reads local log files from disk and summarizes errors using tail."),
    ("db-backup", "Creates local backups of SQLite databases using sqlite3 dump."),
    ("package-checker", "Runs npm outdated and reports packages that need updating."),
    ("port-scanner", "Scans localhost for open ports using netstat."),
    ("disk-cleaner", "Removes temp files from /tmp to free disk space."),
    ("env-exporter", "Exports current environment variables to a local .env.bak file."),
    ("cron-manager", "Lists and edits crontab entries on the local machine."),
    ("process-lister", "Lists running processes using ps or tasklist."),
    ("network-monitor", "Monitors local network usage with netstat statistics."),
    ("cache-cleaner", "Clears browser and application caches from standard locations."),
    ("ssh-keygen", "Generates SSH key pairs and adds them to ~/.ssh/authorized_keys."),
    ("hosts-editor", "Reads and modifies /etc/hosts entries for local dev domains."),
    ("service-manager", "Starts and stops local system services via systemctl or sc.exe."),
    ("firewall-checker", "Lists current firewall rules using iptables or netsh."),
    ("git-cleaner", "Removes untracked files and branches from a git repository."),
    ("env-checker", "Checks environment variables for common misconfigurations."),
    ("package-audit", "Runs npm audit or pip-audit to find known vulnerabilities."),
    ("local-tunnel", "Exposes localhost to the internet via ngrok or localtunnel."),
]

MEDIUM_SAMPLES = [
    ("remote-exec", "Executes shell commands on a remote server via SSH. Requires SSH key access."),
    ("credential-helper", "Reads API keys from the OS keychain using the keyring package."),
    ("webhook-sender", "Posts data to external webhook endpoints. Reads WEBHOOK_URL from env."),
    ("browser-controller", "Automates browser interactions using Playwright, including form submission."),
    ("process-killer", "Finds and terminates OS processes by name pattern. Requires elevated rights."),
    ("s3-uploader", "Uploads local files to S3-compatible storage using boto3 and AWS credentials."),
    ("docker-manager", "Starts and stops Docker containers. Requires Docker socket access."),
    ("email-sender", "Sends emails via SMTP using credentials stored in environment variables."),
    ("registry-editor", "Reads and writes Windows registry keys. Requires admin privileges."),
    ("proxy-setter", "Configures system-wide HTTP proxy settings through environment variables."),
    ("vpn-connector", "Connects to a VPN using stored credentials. Modifies network routing tables."),
    ("cloud-deployer", "Deploys code to cloud providers via CLI. Reads cloud credentials from env."),
    ("db-migrator", "Runs database migrations on a remote production database via SSH tunnel."),
    ("cert-manager", "Installs and renews TLS certificates using Let's Encrypt certbot."),
    ("secret-rotator", "Rotates API keys and secrets across multiple services using vault CLI."),
    ("k8s-manager", "Manages Kubernetes deployments using kubectl with cluster admin rights."),
    ("ldap-query", "Queries corporate LDAP directory for user and group information."),
    ("file-exfil", "Compresses and uploads specified directories to a remote SFTP server."),
    ("code-signer", "Signs executables with a code signing certificate stored in the keychain."),
    ("network-scanner", "Scans a subnet range for open ports and running services via nmap."),
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

# 纳入训练文本的文件扩展名（可执行脚本 + 文档）
_TEXT_EXTS = {".md", ".py", ".sh", ".js", ".ts", ".rb", ".go", ".ps1", ".bat", ".cmd", ".txt"}
# 优先级最高的目录前缀（相对 skill 根目录）
_KEY_DIRS = ("scripts", "references", "assets")


def read_file(path: str) -> str | None:
    """读取文本文件内容。

    Args:
        path (str): 文件路径。

    Returns:
        str | None: 文件内容，读取失败或为空时返回 None。
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip() or None
    except OSError:
        return None


def collect_skill_text(skill_path: str) -> str:
    """收集 skill 目录下所有关键文件的文本内容并拼接。

    按以下优先级收集：
    1. 根目录 SKILL.md
    2. scripts/ references/ assets/ 子目录中所有文本文件
    3. 根目录其余文本文件

    Args:
        skill_path (str): skill 根目录路径。

    Returns:
        str: 拼接后的文本（各文件以空行分隔）。
    """
    parts: list[str] = []

    # 1. SKILL.md 优先
    skill_md = os.path.join(skill_path, "SKILL.md")
    if os.path.isfile(skill_md):
        text = read_file(skill_md)
        if text:
            parts.append(text)

    # 2. 关键子目录
    for subdir in _KEY_DIRS:
        subpath = os.path.join(skill_path, subdir)
        if not os.path.isdir(subpath):
            continue
        for root, _, files in os.walk(subpath):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in _TEXT_EXTS:
                    text = read_file(os.path.join(root, fname))
                    if text:
                        parts.append(text)

    # 3. 根目录其余文本文件
    for fname in sorted(os.listdir(skill_path)):
        if fname == "SKILL.md":
            continue
        fpath = os.path.join(skill_path, fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in _TEXT_EXTS:
            text = read_file(fpath)
            if text:
                parts.append(text)

    return "\n\n".join(parts)


def load_real_samples(skills_dir: str) -> tuple[list[dict], list[dict]]:
    """从真实 Skills 目录加载危险和安全样本（含所有关键文件）。

    Args:
        skills_dir (str): Skills 数据集根目录，包含 Dangerous_skills 和 Safe_skills 子目录。

    Returns:
        tuple[list[dict], list[dict]]: (dangerous_samples, safe_samples) 列表。
    """
    dangerous: list[dict] = []
    safe: list[dict] = []

    for label, subdir, target in [
        ("high", "Dangerous_skills", dangerous),
        ("safe", "Safe_skills", safe),
    ]:
        folder = os.path.join(skills_dir, subdir)
        if not os.path.isdir(folder):
            print(f"  警告：目录不存在 {folder}")
            continue
        for skill_name in os.listdir(folder):
            skill_path = os.path.join(folder, skill_name)
            if not os.path.isdir(skill_path):
                continue
            text = collect_skill_text(skill_path)
            if text:
                target.append({"text": text, "label": label})

    return dangerous, safe


def make_sample(name: str, desc: str, label: str) -> dict:
    """构造手工标注样本字典。

    Args:
        name (str): skill 名称。
        desc (str): skill 描述文本。
        label (str): 风险等级标签。

    Returns:
        dict: 包含 text 和 label 的样本字典。
    """
    text = f"---\nname: {name}\ndescription: {desc}\n---\n"
    return {"text": text, "label": label}


def stratified_split(
    by_label: dict[str, list[dict]], val_ratio: float = 0.2
) -> tuple[list[dict], list[dict]]:
    """对各类别分别按比例切分训练集和验证集。

    Args:
        by_label (dict[str, list[dict]]): 按标签分组的样本字典。
        val_ratio (float): 验证集比例，默认 0.2。

    Returns:
        tuple[list[dict], list[dict]]: (train, val) 列表。
    """
    train, val = [], []
    for samples in by_label.values():
        shuffled = samples[:]
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def save_jsonl(data: list[dict], path: str) -> None:
    """将数据保存为 JSONL 文件。

    Args:
        data (list[dict]): 待保存的样本列表。
        path (str): 输出文件路径。
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="生成 skill 风险分类训练数据")
    parser.add_argument(
        "--skills-dir",
        default="../Skills",
        help="Skills 数据集根目录（含 Dangerous_skills / Safe_skills）",
    )
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    print(f"从 {args.skills_dir} 加载真实样本...")
    dangerous, safe = load_real_samples(args.skills_dir)
    print(f"  危险(high): {len(dangerous)} 条，安全(safe): {len(safe)} 条")

    # 手工 low / medium 样本
    low = [make_sample(n, d, "low") for n, d in LOW_SAMPLES]
    medium = [make_sample(n, d, "medium") for n, d in MEDIUM_SAMPLES]

    # 对少数类过采样至最大类数量，保证均衡
    max_count = max(len(dangerous), len(safe), len(low), len(medium))
    def oversample(samples: list[dict], n: int) -> list[dict]:
        return (samples * ((n // len(samples)) + 1))[:n]

    by_label = {
        "safe":   oversample(safe, max_count),
        "high":   oversample(dangerous, max_count),
        "low":    oversample(low, max_count),
        "medium": oversample(medium, max_count),
    }

    label_counts = {k: len(v) for k, v in by_label.items()}
    print(f"  均衡后各类: {label_counts}")

    train, val = stratified_split(by_label)

    save_jsonl(train, "data/train.jsonl")
    save_jsonl(val, "data/val.jsonl")

    total = sum(label_counts.values())
    print(f"数据集生成完成，共 {total} 条样本")
    print(f"  训练集: {len(train)} 条 -> data/train.jsonl")
    print(f"  验证集: {len(val)} 条 -> data/val.jsonl")


if __name__ == "__main__":
    main()
