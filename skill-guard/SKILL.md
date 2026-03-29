---
name: skill-guard
description: 对已安装的 skill 进行自动安全审计，识别恶意或高风险 skill（如要求下载密码压缩包、运行可执行文件、禁用杀毒软件等）。当安装完新 skill 后触发，或用户主动要求安全检查时激活。输出每个 skill 的风险等级（safe / low / medium / high）。
metadata:
  openclaw:
    emoji: "🛡️"
    always: false
    requires:
      bins: ["python"]
---

# Skill Guard

使用 TF-IDF + XGBoost 模型对已安装的 skill 进行风险评级，识别恶意 skill。

## 首次使用：配置环境

**第一步**：在 `skill-guard/scripts/` 目录下运行环境配置脚本，自动安装依赖并验证模型文件：

```bash
cd skill-guard/scripts
python setup_env.py
```

输出示例：
```
Python: 3.11.0  (/usr/bin/python3)
-------------------------------------------------------
  [OK]  scikit-learn 1.3.0
  [OK]  xgboost 2.0.3
-------------------------------------------------------
  [OK]  模型文件完整

环境配置完成，可运行:
  python main.py
```

## 运行扫描

```bash
cd skill-guard/scripts
python main.py
```

## 风险等级说明

| 等级 | 含义 | 建议操作 |
|------|------|----------|
| `safe` | 无风险，正常 skill | 正常使用 |
| `low` | 低风险，涉及本地文件或系统命令 | 留意权限 |
| `medium` | 中等风险，涉及网络、进程或提权操作 | 审查来源 |
| `high` | **高风险，疑似恶意 skill** | **立即删除** |

发现 `high` 风险时程序以退出码 `2` 退出，可集成到自动化流水线。

## 更新模型

如需用新数据重新训练：

```bash
cd src_model
python 01_generate_data.py   # 处理数据
python 02_train_model.py     # 训练并保存模型
cp -r model ../skill-guard/scripts/
```
