import torch
import joblib
import pandas as pd
import numpy as np
import os
import re
import math
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# --- 配置路径 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DL_PATH = os.path.join(BASE_DIR, "openclaw_model_v1")
MODEL_XGB_PATH = os.path.join(BASE_DIR, "xgboost_security_v2.pkl")
SUSPICIOUS_DIR = r"F:\comp\skill\data\Suspicious_skills"

# --- 加载双路引擎 ---
print("正在启动模型")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DL_PATH)
model_dl = AutoModelForSequenceClassification.from_pretrained(MODEL_DL_PATH)
model_xgb = joblib.load(MODEL_XGB_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dl.to(device).eval()


def calculate_entropy(text):
    if not text: return 0
    counter = Counter(text)
    probs = [c / len(text) for c in counter.values()]
    return -sum(p * math.log2(p) for p in probs)


def extract_stats_features(prompt, code):

    full_text = prompt + code
    lines = code.split('\n')
    code_lower = code.lower()
    prompt_lower = prompt.lower()

    # 1. 危险敏感操作扩展 (api_danger)
    danger_patterns = r"(os\.system|eval|exec|subprocess|requests|socket|urllib|getattr|__import__|curl|wget|chmod|chown|fetch|readfile|readfilesync|fs\.read)"
    api_danger_count = len(re.findall(danger_patterns, code_lower))

    # 2. 持久化特征 (persistence_risk)
    persistence_risk = len(
        re.findall(r"(crontab|cron\.d|systemctl|launchctl|init\.d|nohup|bashrc|profile)", code_lower))

    # 3. 敏感目标访问 (sensitive_targets)
    sensitive_targets = len(
        re.findall(r"(\.env|\.git|id_rsa|passwd|shadow|clawdbot|credentials|token|secret)", code_lower))

    # 4. 数据外泄目标 (exfiltration_sites)
    exfiltration_sites = len(
        re.findall(r"(webhook|requestbin|pipedream|ngrok|mockbin|emailhook|postpayload)", code_lower))

    # 5. 注入词汇与伪装检测 (injection_keywords)
    injection_keywords = len(
        re.findall(r"(ignore|system prompt|override|developer mode|you are now|security|integrity|defender)",
                   prompt_lower))

    feat = {
        "total_len": len(full_text),
        "code_len": len(code),
        "prompt_len": len(prompt),
        "entropy": calculate_entropy(full_text),
        "api_danger": api_danger_count,
        "persistence_risk": persistence_risk,
        "sensitive_targets": sensitive_targets,
        "exfiltration_sites": exfiltration_sites,
        "b64_count": len(re.findall(r'[A-Za-z0-9+/]{40,}', full_text)),
        "hex_count": len(re.findall(r'[0-9a-fA-F]{40,}', full_text)),
        "injection_keywords": injection_keywords,
        "comment_count": len(re.findall(r"(#|//)", code)),
        "line_count": len(lines),
        "long_line_max": max([len(l) for l in lines]) if lines else 0,
    }

    # 按照模型要求的 14 个特征顺序排列
    xgb_cols = [
        'total_len', 'code_len', 'prompt_len', 'entropy', 'api_danger',
        'persistence_risk', 'sensitive_targets', 'exfiltration_sites',
        'b64_count', 'hex_count', 'injection_keywords', 'comment_count',
        'line_count', 'long_line_max'
    ]
    return pd.DataFrame([feat])[xgb_cols]


def unified_scan():
    if not os.path.exists(SUSPICIOUS_DIR):
        print(f"找不到目录: {SUSPICIOUS_DIR}")
        return

    folders = [f for f in os.listdir(SUSPICIOUS_DIR) if os.path.isdir(os.path.join(SUSPICIOUS_DIR, f))]
    print(f"开始扫描 Suspicious 库，共 {len(folders)} 个样本...\n")
    print(f"{'Skill 名称':<25} | {'DL 分数':<8} | {'XGB 分数':<8} | {'最终判定'}")
    print("-" * 80)

    conflicts = []

    for folder in folders:
        full_path = os.path.join(SUSPICIOUS_DIR, folder)

        # 1. 读取内容 (包含 .sh, .bash)
        prompt, code = "", ""
        md_path = os.path.join(full_path, "skill.md")
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8', errors='ignore') as f: prompt = f.read().strip()
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.endswith(('.py', '.js', '.sh', '.bash', '.ts')):
                    with open(os.path.join(root, file), 'r', encoding='utf-8',
                              errors='ignore') as f: code += f"\n# --- File: {file} ---\n" + f.read().strip()

        # 2. 模型 1 推理 (CodeBERT)
        # 获取特殊特征标识
        has_obf = "[HAS_CONFUSION] " if len(re.findall(r'[A-Za-z0-9+/]{40,}', prompt + code)) > 0 else ""
        has_sh = "[DANGER_SHELL] " if len(
            re.findall(r"(crontab|cron\.d|curl.*bash|wget.*bash)", code.lower())) > 0 else ""

        dl_input = f"{prompt} [SEP] {has_obf}{has_sh}{code}"
        inputs = tokenizer(dl_input, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model_dl(**inputs)
            dl_prob = F.softmax(outputs.logits, dim=-1)[0][1].item()

        # 3. 模型 2 推理 (XGBoost)
        stats_df = extract_stats_features(prompt, code)
        xgb_prob = model_xgb.predict_proba(stats_df)[0][1]

        # 4. 融合决策逻辑 (V4 风险优先)
        # 只要任何一个模型报极高危，或探测到窃密关联，就判定为危险
        has_leak_signs = (stats_df['sensitive_targets'].values[0] > 0 and stats_df['exfiltration_sites'].values[0] > 0)

        final_score = (dl_prob * 0.5) + (xgb_prob * 0.5)

        if dl_prob > 0.85 or xgb_prob > 0.85 or has_leak_signs:
            status = "🔴 DANGER"
        elif final_score > 0.4:
            status = "🟡 SUSPICIOUS"
        else:
            status = "🟢 SAFE"

        # 记录模型严重分歧的案例
        if abs(dl_prob - xgb_prob) > 0.5:
            status += " ⚠️ CONFLICT"
            conflicts.append(folder)

        print(f"{folder[:25]:<25} | {dl_prob:.4f} | {xgb_prob:.4f} | {status}")

    print("\n" + "=" * 80)
    print(f"扫描结束。发现冲突样本: {len(conflicts)} 个")
    if conflicts:
        print(f"建议人工审计名单: {conflicts}")
    print("=" * 80)


if __name__ == "__main__":
    unified_scan()