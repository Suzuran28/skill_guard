# Skill Guard 🛡️

对已安装的 skill 进行自动安全审计。使用**规则引擎 + TF-IDF/XGBoost 混合模型**扫描 skill 的所有关键文件，识别恶意或高风险行为，输出风险等级（`safe` / `low` / `medium` / `high`）及命中原因。

## 功能特性

- **混合检测**：规则引擎快速识别已知恶意模式，ML 模型覆盖未知威胁
- **多类型文件扫描**：支持 `.md` / `.py` / `.sh` / `.js` / `.ts` / `.ps1` / `.bat` 等
- **四级风险评估**：从无害到高危，精细分级
- **白名单机制**：可信 skill 一键跳过，减少误报
- **CI/CD 集成**：发现 `high` 风险时以退出码 `2` 退出，可直接接入自动化流水线
- **自动选源安装**：`setup_env.py` 并发探测镜像源，选最快的安装依赖

## 检测的威胁类型

| 类型 | 示例行为 |
|------|----------|
| 反弹/绑定 shell | `netcat -e`, reverse shell 脚本 |
| 数据窃取 | 读取 cookie/token/credential 并外发 |
| 下载并执行远程脚本 | `curl ... \| bash` |
| 禁用安全软件 | disable antivirus/defender/firewall |
| 修改 shell 启动项 | 写入 `.bashrc` / `authorized_keys` |
| 动态代码执行 | `eval()`, `exec()`, base64 解码后执行 |
| 提权操作 | `sudo`, UAC bypass, privilege escalation |
| 持久化计划任务 | `schtasks`, `crontab`, `launchd` |
|...|...|

## 快速开始

下载 `skill-guard` 压缩包，将其发送给 openclaw 即可完成安装

首次使用时，openclaw 会自动运行 `setup_env.py` 安装所需依赖（`scikit-learn`、`xgboost`）

[**下载链接**](https://github.com/Suzuran28/skill_guard/releases) 


## 风险等级说明

| 等级 | 触发场景 | 建议操作 |
|------|---------|----------|
| `safe` | 无可疑行为 | 正常使用 |
| `low` | 访问本地文件、环境变量、系统命令 | 留意权限范围 |
| `medium` | 网络外发、提权、动态执行代码、注册表操作 | 审查来源与内容 |
| `high` | 反弹 shell、禁用安全软件、数据窃取、下载执行远程脚本 | **立即删除** |

## 白名单

已确认安全的 skill 会加入白名单，后续扫描将跳过：


## 环境变量

可使用环境变量配置 skill

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SKILL_GUARD_SKILLS_DIR` | skills 根目录路径 | `~/.openclaw/workspace/skills` |
| `SKILL_GUARD_WHITELIST` | 白名单文件路径 | `<skills_dir>/skill-guard/whitelist.txt` |

## 项目结构

```
skill-guard/
├── SKILL.md                  # Skill 元数据与使用说明
├── whitelist.txt             # 白名单（已信任的 skill 名称）
└── scripts/
    ├── main.py               # 主入口，扫描并输出结果
    ├── setup_env.py          # 环境配置脚本
    ├── model/                # 预训练模型文件
    │   ├── tfidf_vectorizer.pkl
    │   ├── xgboost_classifier.pkl
    │   └── label_encoder.pkl
    └── src/
        ├── security_engine.py  # 混合检测引擎（规则 + ML）
        └── check_file.py       # Skill 目录发现与文件枚举

src_model/                    # 模型训练源码
├── 01_generate_data.py       # 从样本生成训练/验证数据
├── 02_train_model.py         # 训练 TF-IDF + XGBoost 模型
└── 03_scan.py                # 独立扫描脚本
```

## 重新训练模型

如需基于自定义样本重新训练模型：

```bash
cd src_model

# 1. 从 Skills/ 目录生成训练数据
python 01_generate_data.py

# 2. 训练模型
python 02_train_model.py

# 3. 将生成的模型文件复制到 skill-guard
cp model/*.pkl ../skill-guard/scripts/model/
```

## CI/CD 集成

扫描发现 `high` 风险 skill 时程序以退出码 `2` 退出，可直接集成到自动化流水线：

```yaml
# GitHub Actions 示例
- name: Skill Security Scan
  run: python ~/.openclaw/workspace/skills/skill-guard/scripts/main.py
  # 发现高危 skill 时自动失败
```

## License

[MIT](LICENSE) © Suzuran
