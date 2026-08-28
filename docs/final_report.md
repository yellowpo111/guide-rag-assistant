# Fiscal RAG Demo v1.1 项目总结

## 项目背景

项目服务于财政软件操作指南问答。真实问题通常包含菜单、岗位、单据状态、业务对象和跨系统流转，仅靠关键词或相似主题匹配容易返回“相关但不能完成操作”的内容。目标是在私有语料边界内提供可追溯回答，同时建立能解释方案选择和失败原因的工程化 Eval，而不是展示一个只在少量示例上可用的聊天界面。

## 最终系统

v1.1 冻结为单机、单 worker、单进程锁的内网 Assistant：

```text
Router
  |-- RAG: Rewrite -> Guard -> Dense Top-20 -> rerank -> Top-5 -> Generation
  |-- Chat: constrained lightweight conversation
  `-- Out of scope: fixed boundary response
```

数据链路为 29 份私有 Markdown、Markdown-aware `1000/100` chunking、1000 chunks、Qwen 1024 维 embedding、Persistent Chroma 和 DeepSeek `deepseek-v4-flash` generation。Web UI、FastAPI 与 SSE 同源部署；`POST /v1/ask` 作为 JSON RAG 兼容接口保留。

这是一套单轮 workflow，不包含 Agent Loop、工具执行、服务端会话或复杂分布式基础设施。它面向小范围、低频内部使用，不被描述为高并发生产平台。

## 实验方法

项目按以下闭环推进：

```text
Baseline -> Evidence-Centric Eval -> Failure Analysis
         -> Single-variable Ablation -> Adopt / Reject / Supersede
```

V1 50 条开发集用于选择方案；V2 15 条与 V4 16 条用于独立验证；V3 10 条仅诊断同术语、不同业务对象的混淆。Hit@k/MRR 要求 chunk 正文覆盖标注 evidence，避免只靠 source 名称判为命中。

## Ablation 与最终选择

Global Dense V1 baseline 为 Hit@1 `0.62`、Hit@3 `0.88`、Hit@5 `0.92`、MRR `0.75`。加入默认 reranker 后达到 `0.74/0.96/1.00/0.853333`，说明主要改进空间是已召回候选内的排序。

Conservative Rewrite 进一步将 V1 提升至 `0.78/1.00/1.00/0.876667`；Guard frozen replay 的最终 MRR 为 `0.88`。Guard 的价值是保护角色与状态约束，不是制造额外候选。

三个看似合理的方案被明确舍弃：

- 财政操作导向 reranker instruction：V1 Hit@1 `0.68`、Hit@5 `0.96`、MRR `0.816667`，低于默认 instruction。
- Metadata-context reranking：Hit@1 `0.72`、MRR `0.84`，标题前缀干扰正文相关性判断。
- Dense/BM25 RRF Hybrid：Hit@3/5 降至 `0.98`，MRR `0.87`，并引入新的 Top-5 miss。

因此代码可以保留历史实验能力，但线上不同时叠加这些模块。

## Top-20 Recovery

Top-10 曾在 V1 保持 `0.78/1.00/1.00/0.88`，因此一度被采用。V2 随后发现 `v2-010`：正确“账号解冻” evidence 位于 Dense Rank 11，Top-10 截断后 reranker 无法看到；generation 安全说明证据不足，没有把相近的“指标解冻”流程当答案。

仅将 candidate pool 改回 Top-20 后，`v2-010` 恢复为最终 Rank 1。V2 达到 Hit@1 `0.80`、Hit@3/5 `0.933333`、MRR `0.855556`；V4 保持 Hit@1 `0.6875`、Hit@3/5 `1.00`、MRR `0.833333`，与 Top-10 对照无逐题回归。这是采用 Top-20 的直接证据，也说明不能只在开发集上冻结容量参数。

## Generation 与 Assistant 质量

V1 Generation 50 条已由用户人工确认：49 条 `supported/correct`，1 条 `partially_supported/partially_correct`，0 条 `unsupported/incorrect`。这个结果说明当前 prompt 与 Top-5 Context 在开发集上没有暴露重大编造，但不能替代独立 Assistant Eval。

v1.1 已新增 54 条 `assistant-eval-v1`：31 条 V2/V4 知识问答、6 条无答案/证据不足、6 条 chat、6 条明确超范围和 5 条路由边界。它从真实 HTTP/SSE 入口验证 Router、RAG、trace、stream completion 和最终回答，并把自动路由指标与按场景人工 adjudication 分开。

截至本文更新，54 条已通过真实 HTTP/SSE 执行。selected 结果为 54/54 completed，route accuracy 与 macro-F1 均为 `1.0`，RAG trace completion 为 `1.0`，没有知识问题误路由。首次完整 attempt 有 1 条空 generation stream；后续单条 attempt 2 遇到一次连接错误，attempt 3 成功，所有失败记录均被保留。

54 条已由用户逐条确认。31 条可回答 RAG 全部 supported，其中 30 correct、1 partially correct，且 31 条 source trace 均 sufficient；6 条无答案问题全部适当 abstain，其中 5 supported/correct、1 partially supported/correct；6 条 chat、6 条 out-of-scope 和 5 条 routing boundary 均遵守能力边界。最终 Assistant `release_gate: passed`，无 release blocker。

两条 partial 构成可解释的 failure analysis：`assistant-v4-012` 漏掉完整菜单路径；`assistant-noanswer-002` 虽正确拒绝代查实时余额，但没有准确区分助手权限与业务系统查询能力。它们不属于错误业务步骤、unsupported 核心结论或不安全越界。

## 性能与可观察性

请求级 timing recorder 可按实际 route 记录 queue wait、router、rewrite、guard、query embedding、vector search、rerank、RAG preparation、generation TTFT、generation 和 server total。`done.timings_ms` 对旧 SSE 客户端向后兼容。

TTFT 定义为客户端发出请求到首个非空 `delta`，不使用 `start` 事件。性能 runner 固定 8 RAG、4 chat、4 out-of-scope，每题 1 warm-up + 3 measured，共 48 个正式请求；另有 3 组双请求检查单进程锁 queue wait，以及独立冷启动至 ready 测量。

2026-08-26 已在实际内网部署机完成基线：48/48 正式请求完成，首次顺序请求成功率和 SSE completion rate 均为 `1.0`。总体 client TTFT p50/p95 为 `2893/4459 ms`，总耗时 p50/p95 为 `3054/5377 ms`；其中 RAG 的 TTFT p50/p95 为 `3954/4930 ms`，总耗时 p50/p95 为 `4310/5568 ms`。RAG preparation p50/p95 为 `1541/2051 ms`，generation p50/p95 为 `1723/3201 ms`。

3 组双 RAG 请求共 6 个请求全部完成，queue wait p50/p95 为 `2076/4584 ms`，说明当前进程锁会造成明显串行排队。一次冷启动至 `/health/ready` 为 `4307 ms`。这些结果只建立当前硬件、网络和外部模型服务下的 baseline，不是压力测试、并发用户容量证明或生产 SLO，也没有据此引入 Redis、多 worker、WebSocket 或微服务。

## 工程与部署经历

项目已经完成 Windows 公司内网单机部署，Web UI 与 API 由同一 FastAPI 服务提供，访问边界依赖内网隔离和限制网段的防火墙规则。持久化索引避免每次启动重新 embedding；manifest 用 corpus hash 与固定参数阻止索引和代码配置静默漂移。

release verifier 进一步检查 Python/依赖、环境变量名、`1.1.0` 版本、29/1000/1024 索引契约、Top-20/Top-5、私有路径 Git 状态和 eval artifact hash，并生成私有 release manifest。

实际部署发生于 2026-08-25 至 2026-08-26，部署机为 Intel Core i5-10400、约 32 GB RAM 的 Windows 单机。当前由前台 PowerShell 手动启动，尚未配置后台常驻服务。1 名实际测试者已从另一台公司内网电脑通过内网 IP + TCP 8000 完成访问验证；这不构成多用户并发或生产容量证明。

## 已确认的限制

- 私有语料和有限 holdout 不能证明跨组织或跨领域泛化。
- V1 是开发集，不能单独写成生产质量结论。
- V4 Generation 是 Assistant 初审，尚不是用户最终确认。
- source trace 是 Top-5 来源追溯，不是逐句 inline citation。
- 外部模型接口可能超时、失败或发生行为变化。
- 单进程锁会让并发请求排队；双请求 sanity check 已观察到 queue wait p50/p95 为 `2076/4584 ms`，尚未验证并发用户容量。
- API 无应用层鉴权，不能开放到公网或未批准网段。
- 系统不执行真实财政操作，回答仍需由业务人员核验。

## 公开展示版迁移

公开版不直接“脱敏后上传”当前私有数据，而是通过可配置 corpus/eval 路径接入独立 `data_demo/`。pipeline、Web UI、Assistant Eval schema 和性能工具可以复用，但必须：

1. 使用有明确许可证的公开数据或全新模拟数据。
2. 重新构建索引并重新运行 Retrieval、Assistant 和性能评估。
3. 不沿用私有问题、答案、source 名、artifact、截图或私有结果数字。
4. 发布前执行 secret/PII 扫描、Git 历史审查、环境变量检查、截图脱敏和许可证确认。
5. 使用独立结果与 tag，避免与内部 `v1.1.0` 证据混淆。

当前阶段只保留迁移设计，不为了公开版大规模重做语料或架构。

## 面试可讲述的核心

这个项目的重点不是组件数量，而是完整闭环：从 Dense baseline 出发，用证据级指标识别排序问题；通过 reranker/rewrite 改善；用 Guard 控制 rewrite 风险；用 `v2-010` 的 Rank-11 failure 证明 Top-20 的必要性；通过负向 ablation 舍弃 instruction、metadata 和 Hybrid；最后把 Router、SSE、人工 groundedness、阶段耗时、内网部署与 release verification 纳入同一可审计版本。

完整实验表见 [experiments.md](experiments.md)，评估执行与 release gate 见 [evaluation.md](evaluation.md)，部署细节见 [deployment_windows.md](deployment_windows.md)。
