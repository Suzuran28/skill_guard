---
name: skill-guard
description: 对已安装的 skill 进行自动安全审计。扫描 SKILL.md、scripts/、references/ 等所有关键文件，使用规则引擎 + TF-IDF/XGBoost 混合模型识别恶意或高风险 skill（如反弹 shell、禁用安全软件、数据窃取、下载执行远程脚本等），输出每个 skill 的风险等级（safe / low / medium / high）及命中原因。在安装新 skill 后、怀疑某 skill 存在恶意行为、或定期巡检时激活。
metadata:
  openclaw:
    emoji: "🛡️"
    always: false
    requires:
      bins: ["python"]
---

# Skill Guard

混合安全引擎（规则匹配 + TF-IDF/XGBoost 模型）扫描已安装的 skill，识别恶意行为并给出风险等级。

## 首次使用：配置环境

在 `skill-guard/scripts/` 目录下运行配置脚本，自动安装依赖并验证模型文件：

```bash
python ~/.openclaw/workspace/skills/skill-guard/scripts/setup_env.py
```

## 运行扫描

```bash
python ~/.openclaw/workspace/skills/skill-guard/scripts/main.py
```

输出示例：

```
正在加载安全引擎...

发现 4 个待检查 skill

Skill                                    风险等级         置信度
-----------------------------------------------------------------
test_360-translate                       MEDIUM    100.0%
                                             ↳ SKILL.md -> 网络数据外发
                                             ↳ scripts/fetch.py -> 读取敏感环境变量
test_a-stock-analysis-1-0-0              SAFE      100.0%
test_agent-browser-xxx                   HIGH      100.0%
                                             ↳ scripts/install.ps1 -> 反弹/绑定 shell
test_auto-updater-xxx                    HIGH      100.0%
                                             ↳ scripts/update.sh -> 注册表操作
-----------------------------------------------------------------

警告：发现 2 个高风险 skill！
  [HIGH] test_agent-browser-xxx
  [HIGH] test_auto-updater-xxx
```

发现 `high` 风险时程序以退出码 `2` 退出，可集成到自动化流水线。

## 风险等级说明

| 等级 | 触发场景 | 建议操作 |
|------|---------|----------|
| `safe` | 无可疑行为，正常 skill | 正常使用 |
| `low` | 访问本地文件、环境变量、系统命令 | 留意权限范围 |
| `medium` | 网络外发、提权、动态执行代码、注册表操作 | 审查来源与内容 |
| `high` | **反弹 shell、禁用安全软件、数据窃取、下载执行远程脚本** | **立即删除** |

## 白名单

在确定 skill 安全或者风险可控的情况下，可以将其添加到白名单中，避免未来扫描时重复报警：

```bash
echo "\n{skill_name}" >> ~/.openclaw/workspace/skills/skill-guard/whitelist.txt
```

> 注意： 上述的 {skill_name} 需要替换为实际的 skill 名称，且白名单应谨慎使用，确保只添加经过充分审查的 skill