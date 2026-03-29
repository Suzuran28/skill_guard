import os
import json
import re
import random
from sklearn.model_selection import train_test_split

# ================= 配置区 =================
# 指向你存放样本的绝对路径
BASE_DATA_DIR = r"F:\comp\skill\data"

# 输出文件名
OUTPUT_TRAIN = "train.jsonl"
OUTPUT_VAL = "val.jsonl"
OUTPUT_TEST = "test.jsonl"


# ==========================================

def detect_malicious_patterns(text):
    """
    这里负责对于Base64 混淆检测和.sh文件的攻击
    """
    flags = ""

    # 1. 编码混淆检测 (Base64/Hex)
    if re.search(r'[A-Za-z0-9+/]{40,}', text):
        flags += "[HAS_CONFUSION] "

    # 2. 危险 Shell 模式检测
    # 匹配 crontab (持久化), curl/wget | bash (远程执行), chmod +x (赋权)
    shell_danger = r"(crontab|cron\.d|curl.*bash|wget.*bash|chmod\s\+x|nc\s-e|/bin/bash|python\s-c)"
    if re.search(shell_danger, text.lower()):
        flags += "[DANGER_SHELL] "

    return flags


def load_raw_data(subfolder_name, label):
    """
    遍历文件夹提取数据
    """
    target_dir = os.path.join(BASE_DATA_DIR, subfolder_name)
    samples = []

    if not os.path.exists(target_dir):
        print(f"警告：跳过不存在的目录: {target_dir}")
        return samples

    skill_folders = [f for f in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, f))]
    print(f"正在从 {subfolder_name} 读取 {len(skill_folders)} 个文件夹...")

    for skill_name in skill_folders:
        folder_path = os.path.join(target_dir, skill_name)

        # 1. 提取提示词 (skill.md)
        prompt = ""
        md_path = os.path.join(folder_path, "skill.md")
        if os.path.exists(md_path):
            try:
                with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
                    prompt = f.read().strip()
            except Exception as e:
                print(f"读取 {md_path} 出错: {e}")

        # 2. 提取脚本代码
        code = ""
        for root, _, files in os.walk(folder_path):
            for file in files:
                # 对 shell 脚本的支持
                if file.endswith(('.py', '.js', '.sh', '.bash')):
                    file_full_path = os.path.join(root, file)
                    try:
                        with open(file_full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            # 记录文件名，帮助模型理解这是 shell 脚本还是 python
                            code += f"\n# --- File: {file} ---\n" + f.read().strip()
                    except Exception as e:
                        print(f"读取脚本 {file_full_path} 出错: {e}")

        # 只有当文件夹里包含 md 或 代码时才记录
        if prompt or code:
            # 构造输入格式
            # 获取混淆或 Shell 风险标记
            feature_flags = detect_malicious_patterns(prompt + code)

            # 提示词 [SEP] 标记 + 代码逻辑
            # CodeBERT 的双输入结构对这种格式非常敏感
            combined_text = f"{prompt} [SEP] {feature_flags}{code}"

            samples.append({
                "text": combined_text,
                "label": label,
                "metadata": {
                    "origin_folder": subfolder_name,
                    "skill_name": skill_name,
                    "has_code": bool(code),
                    "file_types": "mixed"
                }
            })

    return samples


def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print("开始预处理")

    # 1. 加载数据
    safe_data = load_raw_data("Safe_skills", 0)
    dangerous_data = load_raw_data("Dangerous_skills", 1)

    all_data = safe_data + dangerous_data

    if not all_data:
        print("错误：未读取到任何有效数据！")
        exit()

    all_labels = [d['label'] for d in all_data]
    print(f"\n数据提取完成:")
    print(f" - Safe 样本数: {len(safe_data)}")
    print(f" - Dangerous 样本数: {len(dangerous_data)}")

    # 2. 分层拆分
    train_data, temp_data = train_test_split(
        all_data,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )

    temp_labels = [d['label'] for d in temp_data]
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )

    # 3. 保存结果
    save_jsonl(train_data, OUTPUT_TRAIN)
    save_jsonl(val_data, OUTPUT_VAL)
    save_jsonl(test_data, OUTPUT_TEST)

    print(f"\n数据集已保存：")
    print(f" └── {OUTPUT_TRAIN} | {OUTPUT_VAL} | {OUTPUT_TEST}")