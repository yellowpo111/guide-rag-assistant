# Fiscal RAG Demo 文档索引

`v1.1.0` 是冻结的 RAG 稳定基线；v1.2 归档了 SQLite 使用记录与反馈闭环，v1.3 归档了脱敏 Usage Analytics 与行动队列。当前 `v1.4.0` 保持相同 Router、RAG profile、HTTP API 和 SQLite schema，只增加本机维护人员的只读 Text-to-SQL prototype。

## 阅读路径

| 文档 | 唯一职责 |
|---|---|
| [README](../README.md) | 项目入口、核心结果、快速运行、限制与导航 |
| [architecture.md](architecture.md) | v1.4 active path、SQLite/Chroma 边界、维护分析与 Text-to-SQL 实验平面、冻结 Eval path 与 timing 数据流 |
| [experiments.md](experiments.md) | Retrieval 实验与独立 Text-to-SQL prototype 的变量、指标、决策和 artifact |
| [evaluation.md](evaluation.md) | Retrieval/Assistant/groundedness/性能口径、runner 和 release gate |
| [final_report.md](final_report.md) | 问题、方法、ablation、failure analysis、部署经历与面试叙事 |
| [deployment_windows.md](deployment_windows.md) | Windows 安装、网络边界、as-built 事实、验证和回滚 |
| [knowledge_lifecycle.md](knowledge_lifecycle.md) | Markdown 变更、全量索引发布、validation、切换与失败恢复 |
| [usage_data_loop.md](usage_data_loop.md) | 使用记录、反馈、统计、审核、Eval 晋升、保留和备份 |

## 当前证据状态

- Retrieval V1/V2/V4 已完成，当前 Top-20 指标已冻结。
- V1 Generation 50 条已由用户确认：49 supported/correct，1 partial，0 unsupported/incorrect。
- Assistant Eval 54 条真实 HTTP/SSE selected run 与用户逐条 adjudication 均已完成；自动 route/trace 指标通过，人工 release gate passed 且无 blocker。
- 部署性能 runner、阶段 timing 和真实内网部署机 baseline 已完成。
- `v1.1.0` 的 RAG 与性能证据继续作为冻结基线；v1.2 usage evidence 保持归档且不创建 tag；v1.3 的五工作日 production 试运行门槛仍 pending。v1.4 的合成 Text-to-SQL 结果独立记录，不能替代 production 证据。

## 私有数据边界

公司语料、usage SQLite、原始问答、反馈、Eval questions/evidence、逐题结果、人工复核、性能 raw 数据、release manifest 与错误分析都在被 Git 忽略的 `data_private/`。usage 原始记录默认保留 90 天，且不包含身份、网络标识、Context、chunk 正文或 prompt。公开文档只保留聚合指标和脱敏架构；未来展示版必须使用独立公开/模拟数据重新建索引和评估。
