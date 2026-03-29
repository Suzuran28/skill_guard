import os
import re
import math
import torch
import joblib
import pandas as pd
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class OpenClawSecurityEngine:
    def __init__(self, dl_model_path="./openclaw_model_v1", xgb_model_path="./xgboost_security_v2.pkl"):
        """
        初始化：加载 CodeBERT和 XGBoost
        """
        print("正在初始化模型...")

        # 1. 加载 CodeBERT (语义模型)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(dl_model_path)
            self.model_dl = AutoModelForSequenceClassification.from_pretrained(dl_model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_dl.to(self.device).eval()
        except Exception as e:
            print(f"加载 CodeBERT 失败: {e}")

        # 2. 加载 XGBoost (统计模型)
        try:
            self.model_xgb = joblib.load(xgb_model_path)
        except Exception as e:
            print(f"加载 XGBoost 失败: {e}")

        print(f"就绪。监控算力: {self.device}")

    def _calculate_entropy(self, text):
        """计算信息熵：识别加密/混淆载荷"""
        if not text: return 0
        counter = Counter(text)
        probs = [c / len(text) for c in counter.values()]
        return -sum(p * math.log2(p) for p in probs)

    def _extract_v4_features(self, prompt, code):
        """
        特征提取 V4：重点捕捉窃密行为流和系统级干扰
        """
        full_text = prompt + code
        lines = code.split('\n')
        code_lower = code.lower()
        prompt_lower = prompt.lower()

        # [针对性正则] 敏感文件目标：.env, id_rsa, clawdbot 等
        sensitive_targets = len(
            re.findall(r"(\.env|\.git|id_rsa|passwd|shadow|clawdbot|credentials|token|secret)", code_lower))

        # [针对性正则] 数据外泄地址：webhook, requestbin 等
        exfiltration_sites = len(
            re.findall(r"(webhook|requestbin|pipedream|ngrok|mockbin|emailhook|postpayload)", code_lower))

        # [行为流分析] 既有读取本地文件，又有发起网络请求 (Stealthy Data Leak)
        has_read = 1 if re.search(r"(readfile|readfilesync|fs\.read|open\(|cat\s)", code_lower) else 0
        has_net = 1 if re.search(r"(fetch|axios|https?\.request|post|webhook|curl|wget)", code_lower) else 0
        leak_behavior = 1 if (has_read and has_net) else 0

        # 构造 14 个 XGBoost 必需特征
        feat = {
            "total_len": len(full_text),
            "code_len": len(code),
            "prompt_len": len(prompt),
            "entropy": self._calculate_entropy(full_text),
            "api_danger": len(
                re.findall(r"(os\.system|eval|exec|subprocess|requests|socket|fetch|readfile|chmod|curl|wget)",
                           code_lower)),
            "persistence_risk": len(
                re.findall(r"(crontab|cron\.d|systemctl|launchctl|init\.d|nohup|bashrc|profile)", code_lower)),
            "sensitive_targets": sensitive_targets,
            "exfiltration_sites": exfiltration_sites,
            "b64_count": len(re.findall(r'[A-Za-z0-9+/]{40,}', full_text)),
            "hex_count": len(re.findall(r'[0-9a-fA-F]{40,}', full_text)),
            "injection_keywords": len(
                re.findall(r"(ignore|system prompt|override|developer mode|security|integrity|defender)",
                           prompt_lower)),
            "comment_count": len(re.findall(r"(#|//)", code)),
            "line_count": len(lines),
            "long_line_max": max([len(l) for l in lines]) if lines else 0,
            "leak_behavior": leak_behavior  # 仅供内部逻辑判定使用
        }
        return feat

    def scan_skill(self, skill_folder):
        """
        执行扫描：语义审计 + 统计审计 + 红线判定
        """
        # 1. 读取内容 (支持 Python, JS, TS, Shell)
        prompt, code = "", ""
        md_path = os.path.join(skill_folder, "skill.md")
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8', errors='ignore') as f: prompt = f.read().strip()

        for root, _, files in os.walk(skill_folder):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.sh', '.bash')):
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        # 注入文件名上下文
                        code += f"\n# --- File: {file} ---\n" + f.read().strip()

        if not prompt and not code: return "UNKNOWN", 0, {}

        # 2. 提取特征
        feats = self._extract_v4_features(prompt, code)

        # 3. 执行 XGBoost 推理
        xgb_cols = [
            'total_len', 'code_len', 'prompt_len', 'entropy', 'api_danger',
            'persistence_risk', 'sensitive_targets', 'exfiltration_sites',
            'b64_count', 'hex_count', 'injection_keywords', 'comment_count',
            'line_count', 'long_line_max'
        ]
        xgb_input_df = pd.DataFrame([feats])[xgb_cols]
        xgb_prob = self.model_xgb.predict_proba(xgb_input_df)[0][1]

        # 4. 执行 CodeBERT 推理
        # 加入特殊标记以增强模型感知
        has_sh = "[DANGER_SHELL] " if feats['persistence_risk'] > 0 else ""
        dl_input = f"{prompt} [SEP] {has_sh}{code}"

        inputs = self.tokenizer(dl_input, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model_dl(**inputs)
            dl_prob = F.softmax(outputs.logits, dim=-1)[0][1].item()

        # 5. 最终判定逻辑
        # 策略 A: 只要探测到“读取敏感文件”+“向外发数据”的确定性模式，直接封禁 (红线判定)
        if feats['sensitive_targets'] > 0 and feats['exfiltration_sites'] > 0:
            decision = "🔴 DANGEROUS (Stealthy Data Exfiltration)"
            final_score = 1.0

        # 策略 B: 统计模型已经发现高度可疑 (0.65) 且有泄露迹象
        elif xgb_prob > 0.5 and (feats['sensitive_targets'] > 0 or feats['exfiltration_sites'] > 0):
            decision = "🔴 DANGEROUS (Behavioral Match)"
            final_score = max(dl_prob, xgb_prob)

        # 策略 C: 任何一个模型给出极高危评价
        elif dl_prob > 0.85 or xgb_prob > 0.85:
            decision = "🔴 DANGEROUS (High Confidence)"
            final_score = max(dl_prob, xgb_prob)

        # 策略 D: 灰色地带
        elif dl_prob > 0.4 or xgb_prob > 0.4:
            decision = "🟡 SUSPICIOUS"
            final_score = max(dl_prob, xgb_prob)

        # 策略 E: 安全
        else:
            decision = "🟢 SAFE"
            final_score = (dl_prob + xgb_prob) / 2

        return decision, final_score, {
            "dl_score": dl_prob,
            "xgb_score": xgb_prob,
            "leak_detected": feats['leak_behavior'],
            "sensitive_targets": feats['sensitive_targets']
        }


# ================= 实际执行部分 =================
if __name__ == "__main__":
    # 1. 初始化引擎
    engine = OpenClawSecurityEngine(
        dl_model_path="./openclaw_model_v1",
        xgb_model_path="./xgboost_security_v2.pkl"
    )

    # 2. 设定待扫描的总目录 (存放多个 Skill 的文件夹)
    base_scan_path = r"./test_skills"

    if os.path.exists(base_scan_path):
        # 获取该目录下所有的子文件夹
        skill_folders = [f.path for f in os.scandir(base_scan_path) if f.is_dir()]

        print(f"\n启动模型")
        print(f"目标目录: {os.path.abspath(base_scan_path)}")
        print(f"检测到 {len(skill_folders)} 个待审计 Skill...")
        print("=" * 70)

        # 初始化统计数据
        summary_stats = {
            "DANGEROUS": 0,
            "SUSPICIOUS": 0,
            "SAFE": 0,
            "TOTAL": len(skill_folders)
        }
        danger_list = []

        # 3. 循环遍历每个独立的 Skill
        for skill_path in skill_folders:
            skill_name = os.path.basename(skill_path)
            print(f"正在扫描: [{skill_name}] ...")

            status, score, detail = engine.scan_skill(skill_path)

            # 更新统计信息
            if "DANGEROUS" in status:
                summary_stats["DANGEROUS"] += 1
                danger_list.append(skill_name)
                icon = "❌"
            elif "SUSPICIOUS" in status:
                summary_stats["SUSPICIOUS"] += 1
                icon = "⚠️"
            else:
                summary_stats["SAFE"] += 1
                icon = "✅"

            # 打印单个 Skill 的简报
            print(f"{icon} 判定: {status}")
            print(f"分数: {score:.4f} [语义: {detail['dl_score']:.4f} | 统计: {detail['xgb_score']:.4f}]")
            if detail['leak_detected']:
                print(f"发现窃密行为流 [Read + Net]！")
            print("-" * 70)

        # 4. 生成最终总结报告 (Final Report)
        print("\n" + "#" * 70)
        print("#" * 70)
        print(f" 扫描时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" 总计扫描 Skill 数量: {summary_stats['TOTAL']}")
        print("-" * 30)
        print(f" 🔴 高危拦截 (DANGEROUS):  {summary_stats['DANGEROUS']}")
        print(f" 🟡 疑似威胁 (SUSPICIOUS): {summary_stats['SUSPICIOUS']}")
        print(f" 🟢 安全放行 (SAFE):       {summary_stats['SAFE']}")
        print("-" * 30)

        if danger_list:
            print(f"恶意 Skill 黑名单:")
            for item in danger_list:
                print(f"   - {item}")
        print("#" * 70 + "\n")

    else:
        print(f"路径不存在，请检查: {base_scan_path}")