# Fiscal RAG Demo

面向财政软件操作指南的企业内网智能助手。项目把 29 份私有 Markdown（1000 chunks）组织为可追溯的 RAG 知识库，并提供 Assistant Router、Streaming/SSE、Web UI、离线 Eval 与 Windows 单机部署链路。

`v1.1.0` 已作为 RAG 稳定基线封版，v1.2 使用数据层和 v1.3 Usage Analytics 已归档。当前代码进入 `v1.4.0` Text-to-SQL prototype 阶段：不改变冻结的检索、生成、在线 API 或 SQLite schema，只增加本机维护人员针对脱敏 production 安全视图的自然语言查询实验。

## 解决的问题

财政软件操作资料分散、术语相近、步骤依赖业务对象和单据状态。普通语义检索容易找到同主题但不能直接完成任务的内容。本项目采用 Evidence-Centric Eval 和逐题 Failure Analysis，区分候选召回、rerank 排序、rewrite 约束丢失和 generation 使用证据等不同失败层，而不是只看最终回答是否“像对的”。

## v1.4 Active Profile

```text
用户输入
  -> DeepSeek Assistant Router
     |-- rag
     |    -> Conservative Rewrite -> Rewrite Guard
     |    -> Qwen Embedding -> Persistent Chroma Dense Top-20
     |    -> qwen3-rerank (default instruction) -> Top-5 Context
     |    -> DeepSeek `deepseek-v4-flash` grounded streaming generation + source trace
     |-- chat -> 受限的问候/身份/能力说明 streaming generation
     `-- out_of_scope -> 固定能力边界回复
```

冻结条件：Python 3.13、Markdown-aware chunking `1000/100`、Qwen 1024 维 embedding、29 documents、1000 chunks、Dense Top-20、rerank Top-5、DeepSeek `deepseek-v4-flash`、单 Uvicorn worker、单进程请求锁。路由不确定或失败时回退到 `rag`。

v1.2 在该链路外层新增：`request_id -> private SQLite usage record -> feedback -> human review -> versioned Eval candidate`。v1.3 复用这些记录生成私有 JSON/Markdown Analytics 和脱敏行动队列。v1.4 再增加一个与在线链路隔离的本机 Text-to-SQL prototype，只查询 connection-local 的 production 安全视图。Chroma 仍只负责可重建的向量索引，SQLite 不参与检索。

在线 CLI/API 使用持久化 Chroma；冻结 Retrieval Eval 使用 `InMemoryVectorStore` replay。两条路径的存储实现不同，实验参数与检索策略一致。BM25、Hybrid、metadata-context 和自定义 reranker instruction 代码仅作为历史实验保留，不属于 active profile。

## 使用数据闭环

服务在 `data_private/usage/fiscal_rag_usage.sqlite3` 保存真实 HTTP 请求的原始问题、回答、安全 trace、route、执行终态与阶段耗时，默认保留 90 天。数据完全匿名，不保存 IP、User-Agent、账号、会话标识、Context、chunk 正文、prompt 或上游异常消息。

前端在完整 production 回答后提供正面/负面评价。Assistant Eval 与 performance runner 仅通过 localhost 使用独立 `traffic_kind`，不会进入真实使用统计。私有 CLI 默认生成不含问题、答案、trace 或 source 名的 JSON/Markdown 报告；只有显式 `--include-raw-questions` 才加入精确高频问题。行动队列可按 request ID 导出完整人工审核模板，严格确认的案例再进入 Eval candidate；冻结的 v1 数据集不会被自动修改。

维护人员还可以运行 `query_usage.py --question <自然语言问题>`。模型只看到五个安全视图的 schema 和固定指标语义，不看到真实样例行；生成的单条只读 SQL 经过格式检查、SQLite authorizer、只读 URI、`query_only`、超时和结果行数限制后执行。查询结果不会再次发送给模型。该命令不接入 Assistant API，也不允许查看原始问题、回答、source 名或 review reason；需要原文时仍必须走受控 review 导出流程。

## 已有量化证据

所有 Retrieval 指标均为严格 Evidence-Centric 自动判定；V1 是用于方案选择的开发集，V2/V4 是独立 holdout。

| Split | N | Hit@1 | Hit@3 | Hit@5 | MRR | 口径 |
|---|---:|---:|---:|---:|---:|---|
| V1 Top-20 | 50 | 0.780000 | 1.000000 | 1.000000 | 0.880000 | 开发集，自动 |
| V2 Top-20 | 15 | 0.800000 | 0.933333 | 0.933333 | 0.855556 | holdout，自动 |
| V4 Top-20 | 16 | 0.687500 | 1.000000 | 1.000000 | 0.833333 | holdout，自动 |

V2 中唯一已确认的 Top-10 candidate cutoff failure `v2-010`，正确 evidence 位于 Dense Rank 11；扩大到 Top-20 后恢复为最终 Rank 1。V4 相对 Top-10 没有逐题回归。

V1 Generation 共 50 条已由用户人工确认：49 条 `supported/correct`，1 条 `partially_supported/partially_correct`，0 条 `unsupported/incorrect`。这组结果属于开发集，不应表述为生产质量保证。

Assistant Eval 已通过真实 localhost HTTP/SSE 全量运行和用户逐条确认。54 条 selected result 的 SSE completion、route accuracy、macro-F1 与 RAG trace completion 均为 `1.0`，0 个知识问题误路由。31 条可回答 RAG 全部 supported，其中 30 correct、1 partially correct；6 条无答案问题全部适当 abstain，其中 1 条解释为 partially supported/correct；所有 54 条均遵守能力边界，最终 `release_gate: passed` 且无 blocker。首次 attempt 的空 generation stream 和后续一次上游连接错误均作为历史失败保留。

内网性能基线已于 2026-08-26 在实际部署机完成。48 个正式请求全部完成；client TTFT p50/p95 为 `2893/4459 ms`，总耗时 p50/p95 为 `3054/5377 ms`。3 组双请求仅用于观察单进程锁，queue wait p50/p95 为 `2076/4584 ms`；一次冷启动至 ready 为 `4307 ms`。这些结果是当前硬件和外部模型服务下的基线，不是压力测试或生产 SLO。

v1.2 usage 层加入后于 2026-08-27 在同一部署机复测。最终完整运行的 48 个 measured 请求全部完成；client TTFT p50/p95 为 `2465/4472 ms`，总耗时 p50/p95 为 `2600/5171 ms`，6 个双请求全部完成，冷启动单次测量为 `6134 ms`。此前两轮完整性能运行分别出现 1 次和 4 次上游连接失败并已保留，因此不能只用最终成功轮次声称上游稳定性改善。

v1.4 Text-to-SQL 首轮真实 DeepSeek 合成诊断为 answerable `9/10`、sensitive refusal `2/2`，达到预设最低门槛。唯一 denotation mismatch 是模型把 retrieval trace 存在性误当成 source trace 覆盖；该失败被保留，说明安全 schema 仍需要维护人员理解和复核。真实 usage 数据库 smoke 返回 0 条 production 结果，只证明 Eval/performance 流量隔离有效，不代表已有生产使用成效。

## 快速运行

### 1. 环境与私有数据

使用 Python 3.13 创建虚拟环境，并安装锁定依赖：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

将语料放入被 Git 忽略的 `data_private/corpus/`，从 `.env.example` 创建本机 `.env`，填写 DashScope embedding/rerank 与 DeepSeek 配置。不要提交 `.env`、`data_private/`、问题、答案、source 名或截图。

### 2. 构建或校验索引

```powershell
.\.venv\Scripts\python.exe scripts\build_vector_index.py
```

语料与 manifest 未变化时会直接复用持久化索引。新增、修改、删除或重命名 Markdown 后，停止服务并显式使用 `--rebuild`；命令会在 staging 中全量构建、自动验证，成功后才切换正式索引，失败时保留最后一次成功索引。完整流程见 [Knowledge Base Maintenance](docs/knowledge_lifecycle.md)。

### 3. 启动服务

```powershell
.\.venv\Scripts\python.exe scripts\serve_api.py
```

默认入口：

- Web UI：`http://127.0.0.1:8000/`
- OpenAPI：`http://127.0.0.1:8000/docs`
- Liveness：`GET /health/live`
- Readiness：`GET /health/ready`
- 兼容 JSON RAG：`POST /v1/ask`
- Assistant SSE：`POST /v1/assistant/stream`
- Assistant 反馈：`PUT/DELETE /v1/assistant/feedback/{request_id}`

SSE 使用 `start -> route -> trace? -> delta* -> done`；`done` 向后兼容地增加 `timings_ms`。TTFT 定义为客户端发出请求到首个非空 `delta`，不把即时 `start` 算作首 token。

也可直接使用 CLI：

```powershell
.\.venv\Scripts\python.exe scripts\ask_rag.py
```

## 验证与评估

本地回归：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify_release.py --allow-dirty
```

全新 clone 可以直接运行 `pytest`：release contract 检查会在需要临时 SQLite fixture 时自行创建被 Git 忽略的空 `data_private/`。完整 `verify_release.py` 仍需要本机私有语料、索引、release evidence 和模型配置，不能以空公开 clone 代替正式发布验证。

`verify_release.py` 校验 Python、依赖、必需配置、版本号、冻结 RAG 参数、usage/Analytics schema、私有路径 Git 状态、语料 hash 与索引 manifest；它还验证 Text-to-SQL schema/prompt identity、production-only 安全视图、原表/系统表/写操作拒绝和数据库文件不变，并在 `data_private/releases/1.4.0/` 生成不覆盖旧文件的私有 manifest。manifest 会记录最新一次明确标识为 synthetic 的 12 条 DeepSeek 实验结果；该结果不是 production 结论，也不替代 v1.3 的五工作日人工运营门槛。

Assistant Eval、人工 adjudication、性能 runner 和冷启动命令见 [评估说明](docs/evaluation.md)。

## 工程边界

- 当前 source trace 是 Top-5 ranked source/section/subsection，不是逐句 inline citation。
- 请求是单轮的；没有 Agent Loop、工具调用或服务端会话。
- Text-to-SQL 是独立 prototype，不是 Agent，也不是普通用户功能；生成 SQL 与结果必须由维护人员按分析用途复核。
- API 自身不鉴权，内网部署依赖网络隔离与仅允许批准网段的 Windows 防火墙规则。
- 当前只验证了 1 名实际测试者；单 worker 和单进程锁会让同时到达的请求排队，尚未验证可支持的并发用户数。
- 外部 embedding、rerank、rewrite、generation 服务仍可能超时或失败。
- 私有 corpus 和有限 holdout 不能证明跨组织泛化能力。

## 文档

- [系统架构](docs/architecture.md)：描述 v1.4 active path、SQLite/Chroma 边界、维护分析与 Text-to-SQL 实验平面、冻结 Eval path 与 timing 数据流。
- [实验登记表](docs/experiments.md)：实验变量、结果、采用状态和私有 artifact。
- [评估说明](docs/evaluation.md)：Retrieval、Assistant、groundedness/source trace 与性能方法。
- [项目总结](docs/final_report.md)：问题、方法、ablation、failure analysis、部署与局限。
- [Windows 内网部署](docs/deployment_windows.md)：安装、验证、网络边界、as-built 字段和回滚。
- [Knowledge Base Maintenance](docs/knowledge_lifecycle.md)：Markdown 变更、全量 rebuild、validation、安全切换和失败恢复。
- [文档索引](docs/system_overview.md)：各文档的职责边界。
- [使用数据闭环](docs/usage_data_loop.md)：SQLite 边界、反馈、报告、审核、Eval 晋升和保留策略。

## 公开展示版规划

未来公开版通过可配置 corpus/eval 路径接入独立 `data_demo/`，复用同一 pipeline、Web UI 和 Eval schema。公开版必须使用有许可证的公开数据或全新模拟数据，重新建索引并重跑全部指标；不得沿用私有问题、答案、source 名、实验详情、截图或本 README 中的私有结果数字。发布前还需完成 secret/PII 扫描、Git 历史审查、环境变量检查、截图脱敏和数据许可证确认，并使用独立结果与 tag。
