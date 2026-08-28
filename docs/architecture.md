# v1.4 系统架构

本文描述 `v1.4.0` active architecture。RAG profile 继续使用 `v1.1.0` 冻结配置，SQLite schema 继续使用 v1.2 的五张关系表，v1.3 Usage Analytics/Failure Operations 继续作为固定维护能力；v1.4 只增加隔离的本机 Text-to-SQL prototype。实验取舍见 [实验登记表](experiments.md)，指标口径见 [评估说明](evaluation.md)。

## 运行边界

系统面向公司内网低频试用，采用单台 Windows 主机、单 Uvicorn worker 和进程内请求锁。Web UI、API、Assistant Router、RAG pipeline 与 SSE 都在同一 Python 进程中；embedding、rerank、rewrite 和 generation 通过外部兼容 API 调用。当前只有 1 名实际测试者完成跨电脑访问验证，尚未验证并发用户容量。

系统不包含多 Agent、Agent Loop、MCP、WebSocket、Redis、任务队列、微服务或服务端会话。API 自身不鉴权，访问边界由公司网络隔离和 Windows 防火墙控制。

## 在线 Active Path

```text
Browser / HTTP client
  -> POST /v1/assistant/stream
  -> FastAPI + request ID + private SQLite started record
  -> single-process lock
  -> Assistant Router
     |-- rag
     |    -> conservative rewrite
     |    -> constraint-preservation guard
     |    -> query embedding
     |    -> Persistent Chroma dense Top-20
     |    -> qwen3-rerank, default instruction
     |    -> Top-5 context + safe source metadata
     |    -> grounded DeepSeek streaming generation
     |-- chat
     |    -> constrained DeepSeek streaming generation
     `-- out_of_scope
          -> fixed boundary response
  -> SSE start / route / trace? / delta* / done | error
  -> SQLite terminal status / safe trace / timing / answer
  -> optional positive or negative feedback
```

Router 仅输出 `rag`、`chat` 或 `out_of_scope`。财政业务问题和不确定输入进入 `rag`；路由调用失败或输出非法也回退到 `rag`，避免用闲聊路径替代知识回答。`chat` 只处理问候、致谢、告别、身份和能力说明。

`POST /v1/ask` 继续提供非流式 JSON RAG 兼容接口；CLI 直接使用同一 persistent RAG builder。服务端固定 Dense Top-20 与 Context Top-5，客户端不能覆盖检索参数。

只有 HTTP `/v1/assistant/stream` 与 `/v1/ask` 进入 usage store；开发 CLI 不采集。Assistant Eval 与性能 runner 必须从 localhost 访问，API 才接受其 `traffic_kind`；远程标记按 production 处理。完整设计见 [使用数据闭环](usage_data_loop.md)。

## 两类持久化

Persistent Chroma 保存由私有 corpus 派生的 embedding 与 chunk metadata，可以从语料重建。`data_private/usage/fiscal_rag_usage.sqlite3` 保存不可重建的请求、回答、反馈和人工审核，使用标准关系表和 90 天原始数据保留期。两者目录、schema、备份和清理互相独立，不共享 Chroma 内部数据库。

## 维护分析平面

Usage Analytics 不经过普通 Assistant Router，也不新增 HTTP 管理接口。维护人员在部署机运行 `summarize_usage.py`，从同一 repository 读取 production 记录并生成私有 JSON/Markdown；默认报告只含聚合、request ID 和稳定错误元数据。需要查看原始问题、回答和 trace 时，再以 request ID 显式运行 `export_usage_review.py`。该边界避免在当前无应用层鉴权的 8000 端口暴露跨用户使用数据。

Text-to-SQL 与固定 Analytics 并列位于维护分析平面：

```text
maintainer question
  -> isolated DeepSeek SQL prompt (safe schema, no sample rows)
  -> strict {"sql": ...} or stable refusal
  -> shape validation
  -> SQLite mode=ro + connection-local TEMP VIEW
  -> query_only + authorizer + resource limits
  -> bounded deterministic result
  -> private JSON + Markdown
```

五个 TEMP VIEW 固定过滤 `traffic_kind='production'`，只暴露请求元数据、timing、二元 feedback、脱敏 review 元数据和每请求 source 数量。原始问题、回答、retrieval/rewrite query、source 名、review reason、expected answer 和 evidence 不进入视图或 prompt。数据库查询结果不再传回模型。v1.3 Python Analytics 继续是 p50/p95 等固定指标的权威实现；prototype 只承诺 SQL 能直接表达的 count、avg/min/max、列表和 JOIN，并要求维护人员审核生成 SQL 与结果。

执行器以 `mode=ro` 打开已存在数据库，先建立连接内视图，再启用 `query_only`。SQLite authorizer 最终限制只能读取安全视图和最小函数 allowlist；文本检查只负责提前拒绝注释、多语句、非 SELECT 和递归 CTE。执行还有约 2 秒进度中断、SQLite limit、默认 50 行和硬上限 100 行。该能力没有 raw mode；原文查看继续走 `export_usage_review.py --request-id`。

## 索引与固定参数

```text
29 private UTF-8 Markdown files
  -> Markdown header-aware split
  -> recursive split for oversized sections
  -> chunk_size=1000, chunk_overlap=100
  -> 1000 LangChain Documents with source/section/subsection
  -> Qwen 1024-dimensional document embeddings
  -> Chroma PersistentClient + private manifest
```

manifest 绑定 corpus hash、embedding model、embedding dimension、chunk 参数、文档数和 chunk 数。索引存在且 current 时，启动和校验不会重新调用 document embedding API；语料变化后通过显式 staging 全量重建、validation 和目录切换发布。仅工作 corpus 存在未发布变化时，在线服务继续最后成功索引并记录安全告警；模型、chunk profile、schema 或 collection 不兼容仍阻断启动。维护流程见 [Knowledge Base Maintenance](knowledge_lifecycle.md)。

在线 active profile：

| 项目 | 固定值 |
|---|---|
| Python | 3.13 |
| Documents / chunks | 29 / 1000 |
| Chunking | 1000 / 100 |
| Embedding | Qwen，1024 维 |
| Vector store | Persistent Chroma |
| Query preparation | Conservative Rewrite + Guard |
| Candidate retrieval | Global Dense Top-20 |
| Reranker | `qwen3-rerank`，default instruction，`page_content` |
| Generation context | Top-5 |
| Generator | DeepSeek `deepseek-v4-flash` |
| Concurrency | single worker + process lock |

## Rewrite、Guard 与 Retrieval

Rewrite 只重述原问题已有的业务对象、动作、状态和限制，不生成答案。Guard 检查可识别角色和高风险状态是否在 rewrite 中被遗漏；发现遗漏时使用原问题检索。Guard 不增加候选、不修改分数，也不是第二个检索器。

Dense retrieval 对全部 chunks 做全局语义搜索，没有 domain filtering、metadata filtering 或 retrieval routing。Reranker 只对 Dense Top-20 候选重新排序，再选择 Top-5。若正确 evidence 不在 Top-20，reranker 无法恢复。

source、section、subsection 用于追溯、UI 展示和人工审核。它们不参与 active reranker 输入，也不代表逐句 citation。

## Streaming 与 Timing

SSE 契约：

```text
start -> route -> trace (仅 RAG) -> delta ... -> done
                                        `-> error（流建立后的失败）
```

`done` 保留原有 `route`，并新增向后兼容的 `timings_ms`。现有客户端可忽略未知字段。TTFT 的客户端口径是发出请求到首个非空 `delta`；`start` 不计作首 token。

每个请求使用隔离的 timing recorder。服务端可能记录：

- `queue_wait`
- `router`
- `rewrite`
- `guard`
- `query_embedding`
- `vector_search`
- `dense_retrieval`
- `rerank`
- `rag_preparation`
- `generation_ttft`
- `server_ttft`
- `generation`
- `server_total`

字段按实际 route 和执行阶段出现，不要求每条请求拥有全部字段。`server_ttft` 从进入服务到首个非空 delta，包含 queue 与准备阶段；`generation_ttft` 只描述 generation 调用开始后的首 delta。

## 冻结 Eval Path

历史 V1-V4 Retrieval Eval 为了复现实验，继续使用固定语料与 `InMemoryVectorStore`：

```text
frozen case + evidence
  -> frozen/live rewrite as specified by experiment
  -> guard replay as specified by experiment
  -> in-memory global dense candidates
  -> same reranker profile
  -> strict Evidence-Centric Hit@k / MRR
```

Persistent Chroma 是在线工程存储方式，不是新的 Retrieval Strategy。在线 path 与冻结 eval path 不能在文档中混写为同一种存储实现。

Assistant Eval 则必须穿过真实 `POST /v1/assistant/stream`，覆盖 Router、RAG、SSE、trace 和最终回答，不允许绕过 HTTP 层直接调用内部对象。

## 历史实验模块

以下代码保留用于审计和复现实验，但不在 v1.4 active path：

- Dense-only baseline。
- 财政操作导向 reranker instruction。
- metadata-context reranking。
- BM25 retrieval 与 Dense/BM25 RRF Hybrid。
- Top-10 candidate profile。

这些模块的存在不表示线上同时启用。具体 adopted/rejected/superseded 状态见 [实验登记表](experiments.md)。

## 数据安全

语料、Eval、usage SQLite、逐题结果、人工复核、性能 raw JSONL、release manifest 和 Failure Analysis 都在被 Git 忽略的 `data_private/`。stdout 日志仍只记录 request ID、route、状态、耗时与异常类型。原始问题和回答只进入受 ACL 与保留期控制的 usage SQLite，不进入普通日志；Context、文档正文、prompt、密钥、身份和网络标识不持久化。
