import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report

# --- 路径硬编码（根据你的截图绝对对齐） ---
# 获取当前脚本所在目录 (即 F:\comp\skill\src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型文件夹就在 src 下
MODEL_PATH = os.path.join(BASE_DIR, "openclaw_model_v1")
# 测试数据就在 src 下
TEST_FILE = os.path.join(BASE_DIR, "test.jsonl")

print(f"检查路径...")
print(f" - 模型路径: {MODEL_PATH}")
print(f" - 测试文件: {TEST_FILE}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型文件夹，请确认路径: {MODEL_PATH}")
if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(f"找不到测试集文件: {TEST_FILE}")

# --- 1. 加载模型与分词器 ---
print("正在加载模型...")
# 解决 Windows 路径可能触发的 HFValidationError，使用绝对路径
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- 2. 加载测试数据 ---
test_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        test_samples.append(json.loads(line))

print(f"开始盲测，总样本数: {len(test_samples)}")

# --- 3. 执行推理 ---
y_true = []
y_pred = []

for i, sample in enumerate(test_samples):
    # 使用与训练一致的 512 长度
    inputs = tokenizer(
        sample["text"],
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # 获取得分最高的类别
        prediction = torch.argmax(outputs.logits, dim=-1).item()

    y_true.append(sample["label"])
    y_pred.append(prediction)

    if (i + 1) % 50 == 0:
        print(f"已处理 {i + 1} 个样本...")

# --- 4. 生成报告 ---
print("\n" + "=" * 50)
print("检测结果")
print("=" * 50)

# 如果你的标签对应是 0: Safe, 1: Dangerous
print(classification_report(
    y_true,
    y_pred,
    target_names=["Safe", "Dangerous"],
    digits=4
))
print("=" * 50)