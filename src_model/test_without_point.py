import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ================= 配置路径 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "openclaw_model_v1")
DATA_DIR = r"F:\comp\skill\data"  # 原始样本库


# ===========================================

def get_raw_content(folder_path):
    """
    纯原始提取：不加任何人工标记，只做最基础的文件读取
    """
    prompt = ""
    md_path = os.path.join(folder_path, "skill.md")
    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
            prompt = f.read().strip()

    code = ""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.py', '.js')):
                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                    code += f"\n{f.read().strip()}"

    # 注意：这里我们只用 [SEP] 分隔符（因为模型结构需要它来区分输入段）
    # 但我们彻底去掉了 [HAS_CONFUSION] 那个正则标记
    return f"{prompt} [SEP] {code}"


def main():
    # 1. 加载模型
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. 准备盲测样本
    categories = {"Safe_skills": 0, "Dangerous_skills": 1}
    results = []

    print(f"开始裸测...")
    print(f"{'来源库':<15} | {'文件夹名':<25} | {'预测结果':<12} | {'可信度'}")
    print("-" * 75)

    for cat_name, true_label in categories.items():
        cat_path = os.path.join(DATA_DIR, cat_name)
        if not os.path.exists(cat_path): continue

        folders = [f for f in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, f))]

        # 每个库随机抽测 20 个，或者你可以去掉切片测全部
        for folder_name in folders[:20]:
            full_path = os.path.join(cat_path, folder_name)
            raw_text = get_raw_content(full_path)

            # 推理
            inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                conf = probs[0][pred].item()

            res_str = "DANGEROUS" if pred == 1 else "SAFE"
            # 标记是否判断正确
            is_correct = "✅" if pred == true_label else "❌ WRONG"

            print(f"{cat_name:<15} | {folder_name[:25]:<25} | {res_str:<12} | {conf:.2%} {is_correct}")


if __name__ == "__main__":
    main()