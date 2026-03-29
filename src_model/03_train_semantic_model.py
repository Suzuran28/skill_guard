import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import evaluate
import numpy as np

# ================= 1. 配置与路径 =================
MODEL_NAME = "microsoft/codebert-base"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
OUTPUT_DIR = "./openclaw_model_v1"

# ================= 2. 加载数据集 =================
print("正在加载数据集...")
dataset = load_dataset('json', data_files={'train': TRAIN_FILE, 'validation': VAL_FILE})

# ================= 3. 分词处理=================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 关键：将我们在预处理中定义的自定义 Token 加入词表
# 这样模型就能理解 [HAS_CONFUSION] 代表混淆特征
special_tokens_dict = {'additional_special_tokens': ['[HAS_CONFUSION]', '[SEP]']}
tokenizer.add_special_tokens(special_tokens_dict)


def tokenize_function(examples):
    # max_length=512 是 CodeBERT 的上限。
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")


print("正在进行文本分词...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ================= 4. 定义评估指标  =================
# F1 和 Recall 比 Accuracy 更重要 主要关注recall值
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# ================= 5. 自定义加权损失 Trainer =================
# 核心逻辑：因为 Safe(800) 多于 Dangerous(500)，我们给 Dangerous 设置更高权重 (1.6倍)
# 这样模型如果漏报一个恶意 Skill，受到的惩罚会更大。
class BlueTeamTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 设置权重: [Safe_Weight, Dangerous_Weight] -> 1:1.6
        # 确保权重张量与模型在同一个设备上
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.6]).to(model.device))

        # 计算损失
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ================= 6. 初始化模型 =================
print("正在初始化 CodeBERT 模型...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
# 必须调整 Embedding 大小，因为我们增加了自定义特殊 Token
model.resize_token_embeddings(len(tokenizer))

# ================= 7. 训练参数设置 =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    logging_steps=20,
    push_to_hub=False,
    report_to="none"
)

# ================= 8. 启动训练 =================
trainer = BlueTeamTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("启动训练...")
trainer.train()

# ================= 9. 保存模型 =================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成！模型已保存至: {OUTPUT_DIR}")