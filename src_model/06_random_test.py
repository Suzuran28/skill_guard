import os
import random
import pandas as pd
from security_engine_core import OpenClawSecurityEngine  # 确保你的 engine.py 是 V4 版本

# ================= 配置区 =================
DATA_DIR = r"F:\comp\skill\data"
SAFE_DIR = os.path.join(DATA_DIR, "Safe_skills")
DANGER_DIR = os.path.join(DATA_DIR, "Dangerous_skills")

SAMPLE_SIZE = 500


# ==========================================

def run_battle_test():
    # 1. 初始化 V4 引擎
    # 这里的路径需指向你训练好的模型权重
    engine = OpenClawSecurityEngine(
        dl_model_path="./openclaw_model_v1",
        xgb_model_path="xgboost_security_v2.pkl"
    )

    # 2. 准备随机样本
    all_safe = [os.path.join(SAFE_DIR, f) for f in os.listdir(SAFE_DIR) if os.path.isdir(os.path.join(SAFE_DIR, f))]
    all_danger = [os.path.join(DANGER_DIR, f) for f in os.listdir(DANGER_DIR) if
                  os.path.isdir(os.path.join(DANGER_DIR, f))]

    sampled_safe = random.sample(all_safe, min(SAMPLE_SIZE, len(all_safe)))
    sampled_danger = random.sample(all_danger, min(SAMPLE_SIZE, len(all_danger)))

    test_queue = []
    for path in sampled_safe: test_queue.append((path, "SAFE"))
    for path in sampled_danger: test_queue.append((path, "DANGEROUS"))

    # 打乱顺序模拟真实流量
    random.shuffle(test_queue)

    print(f"\n 启动模型，共抽样 {len(test_queue)} 个样本...")
    print(f"{'Skill 文件夹':<30} | {'真实':<6} | {'模型判定':<25} | {'得分'} | {'状态'}")
    print("-" * 110)

    stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    redline_hits = 0

    for folder_path, ground_truth in test_queue:
        folder_name = os.path.basename(folder_path)

        # 执行 V4 扫描
        # 返回: decision (str), score (float), detail (dict)
        decision, score, detail = engine.scan_skill(folder_path)

        # 判定映射
        is_malicious_pred = "DANGEROUS" in decision
        is_malicious_real = (ground_truth == "DANGEROUS")

        # 统计分析
        status_icon = ""
        if is_malicious_real and is_malicious_pred:
            status_icon = "✅ HIT"
            if "Targeted" in decision or "Behavioral" in decision:
                status_icon += " (红线触发 ⚡)"
                redline_hits += 1
            stats["TP"] += 1
        elif not is_malicious_real and not is_malicious_pred:
            status_icon = "✅ PASS"
            stats["TN"] += 1
        elif not is_malicious_real and is_malicious_pred:
            status_icon = "❌ FP (误报)"
            stats["FP"] += 1
        elif is_malicious_real and not is_malicious_pred:
            status_icon = "🔥 FN (漏报)"
            stats["FN"] += 1

        # 打印详细行：包含 DL/XGB 原始分对比
        # 注意：这里 detail 的键名需与 V4 engine.py 保持一致
        dl_s = detail.get('dl_score', 0)
        xgb_s = detail.get('xgb_score', 0)

        print(f"{folder_name[:30]:<30} | {ground_truth[:4]:<6} | {decision[:25]:<25} | {score:.3f} | {status_icon}")

    # 3. 输出总结报告
    print("\n" + "=" * 110)
    print(" 模型总结")
    print("=" * 110)
    total = len(test_queue)
    accuracy = (stats["TP"] + stats["TN"]) / total
    recall = stats["TP"] / (stats["TP"] + stats["FN"]) if (stats["TP"] + stats["FN"]) > 0 else 0
    precision = stats["TP"] / (stats["TP"] + stats["FP"]) if (stats["TP"] + stats["FP"]) > 0 else 0

    print(f"  [+] 成功拦截恶意 Skill (TP): {stats['TP']}  (其中红线强制拦截: {redline_hits} 次)")
    print(f"  [+] 正常放行合法 Skill (TN): {stats['TN']}")
    print(f"  [-] 误报误伤合法 Skill (FP): {stats['FP']}")
    print(f"  [!] 严重漏报恶意 Skill (FN): {stats['FN']}")
    print("-" * 40)
    print(f"  总体准确率 (Accuracy):  {accuracy:.2%}")
    print(f"  威胁捕获率 (Recall):    {recall:.2%}")
    print(f"  判定精确率 (Precision): {precision:.2%}")
    print("=" * 110)

    if stats["FN"] > 0:
        print(" 检测到漏网之鱼！请手动审查 FN 样本，并将它们加入 Dangerous 库进行重训。")
    elif stats["FP"] > 5:
        print(" 误报数略高，可能由于正常脚本代码过于复杂，建议调高 engine.py 中的拦截阈值。")
    else:
        print(" 当前十分稳定")


if __name__ == "__main__":
    run_battle_test()