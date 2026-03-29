import os
import re
import math
import pandas as pd
from collections import Counter

# 配置路径
BASE_DATA_DIR = r"F:\comp\skill\data"


def calculate_entropy(text):
    """计算香农熵：混淆/加密文本的熵值通常极高"""
    if not text: return 0
    counter = Counter(text)
    probs = [c / len(text) for c in counter.values()]
    return -sum(p * math.log2(p) for p in probs)


def get_stats(folder_path, label):
    prompt = ""
    code = ""

    # 1. 读取提示词
    md_path = os.path.join(folder_path, "skill.md")
    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
            prompt = f.read()

    # 2. 读取脚本 -增加 .sh 和 .bash
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.py', '.js', '.sh', '.bash')):
                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                    code += f.read()

    full_text = prompt + code
    lines = code.split('\n')
    code_lower = code.lower()
    prompt_lower = prompt.lower()

    # --- 构造特征向量 ---
    features = {
        "label": label,
        "total_len": len(full_text),
        "code_len": len(code),
        "prompt_len": len(prompt),
        "entropy": calculate_entropy(full_text),

        # [维度1] 通用敏感 API (Python/JS/Shell 混合)
        # 增加了 curl, wget, chmod 等系统级命令
        "api_danger": len(re.findall(
            r"(os\.system|eval|exec|subprocess|requests|socket|urllib|getattr|__import__|curl|wget|chmod|chown)",
            code_lower)),

        # [维度2] 持久化特征
        # 检测是否尝试修改定时任务或系统启动项
        "persistence_risk": len(re.findall(
            r"(crontab|cron\.d|systemctl|launchctl|init\.d|nohup|bashrc|profile)",
            code_lower)),

        # [维度3] 敏感目标访问
        # 检测是否尝试读取隐私文件
        "sensitive_targets": len(re.findall(
            r"(\.env|\.git|id_rsa|passwd|shadow|clawdbot|credentials|token)",
            code_lower)),

        # [维度4] 数据外泄目标
        # 检测是否存在黑客常用的数据回传地址
        "exfiltration_sites": len(re.findall(
            r"(webhook\.site|requestbin|pipedream|ngrok|mockbin|emailhook)",
            code_lower)),

        # [维度5] 混淆纹理特征
        "b64_count": len(re.findall(r'[A-Za-z0-9+/]{40,}', full_text)),
        "hex_count": len(re.findall(r'[0-9a-fA-F]{40,}', full_text)),

        # [维度6] 提示词注入与伪装
        # 增加 security, integrity 等词汇，用于识破伪装成安全工具的木马
        "injection_keywords": len(re.findall(
            r"(ignore|system prompt|override|developer mode|you are now|security|integrity|defender)",
            prompt_lower)),

        # [维度7] 代码结构特征
        "comment_count": len(re.findall(r"(#|//)", code)),
        "line_count": len(lines),
        "long_line_max": max([len(l) for l in lines]) if lines else 0,
    }
    return features


def main():
    all_stats = []
    # 扫描两个库
    for cat, label in {"Safe_skills": 0, "Dangerous_skills": 1}.items():
        cat_path = os.path.join(BASE_DATA_DIR, cat)
        if not os.path.exists(cat_path):
            print(f"找不到目录: {cat_path}")
            continue

        print(f"正在从 {cat} 提取统计特征...")
        for folder in os.listdir(cat_path):
            folder_path = os.path.join(cat_path, folder)
            if os.path.isdir(folder_path):
                all_stats.append(get_stats(folder_path, label))

    df = pd.DataFrame(all_stats)
    # 保存 CSV
    output_file = "skill_stats_v2.csv"
    df.to_csv(output_file, index=False)
    print(f"\n特征提取完成！")
    print(f" - 存入文件: {output_file}")
    print(f" - 总计样本: {len(df)}")
    print(f" - 包含特征数: {len(df.columns) - 1}")


if __name__ == "__main__":
    main()