
## 文件说明

### 1. 数据准备
*   **01_dataset_splitter.py**: 扫描 `Safe_skills` 和 `Dangerous_skills` 目录，进行清洗并按照 8:1:1 比例生成 `train.jsonl`, `val.jsonl`, `test.jsonl`。支持对 `.sh` 脚本的扫描。
*   **02_feature_extractor.py**: 提取代码的信息熵、敏感 API 频率、持久化风险、数据外泄目标等 14 维统计特征，生成 `skill_stats_v2.csv`。

### 2. 模型训练
*   **03_train，test_codeBERT与test_without_point.py**: 基于 CodeBERT 的深度学习微调脚本。训练识别提示词与代码之间的逻辑冲突,主要针对skill.md与常见恶意代码的处理。
*   **04_train_statistical_model.py**: 基于 XGBoost 的统计模型训练脚本。利用 GPU 加速，针对高熵混淆和异常 API 调用进行检测。

### 3. 推理引擎 
*   **security_engine_core.py**: 封装了双路推理逻辑。采用“风险优先”决策，使用红线机制（如检测到“读私密文件+外发网络”将直接拦截，对此可能会对于一些对于社交媒体的skill进行误杀）。

### 4. 测试工具
*   **05_unified_gateway_scan.py**: 专门用于审计 `Suspicious_skills` 库，自动识别两个模型冲突（Conflict）样本并输出判定报告。
*   **06_random_test.py**: 从样本库中随机抽取样本进行对比，生成包含召回率（Recall）和拦截率的结果。

### 5. 测试与基准
*   **test_codeBERT.py**: 语义模型专项评估工具。
*   **test_without_point.py**: 逻辑裸测。在不添加任何人工预处理标记的情况下，测试模型对原始代码逻辑的识别能力。

##  防御逻辑 
1.  **语义层**: CodeBERT 识别提示词注入与话术伪装。
2.  **统计层**: XGBoost 识别 Base64/Hex 混淆及 API 异常堆叠。
3.  **行为层**: 针对“窃密流”的硬红线检测（读取 `.env` / `id_rsa` + 访问 `webhook` = 100% 恶意）。
4.  **全文件**: 完整覆盖 `.py`, `.js`, `.ts`, `.sh`, `.bash` 文件。

##  数据文件说明
*   `openclaw_model_v1/`: 存放训练好的 CodeBERT 模型权重。
*   `xgboost_security_v2.pkl`: 存放训练好的 XGBoost 特征模型。
*   `*.jsonl`: 预处理后的结构化文本数据。
*   `skill_stats_v3.csv`: 提取的统计特征数据集。