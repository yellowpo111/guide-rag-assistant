# v1.4 使用数据闭环、Analytics 与只读查询实验

本文定义 v1.2 已归档的真实请求持久化、用户反馈、Failure Analysis 和 Eval 候选晋升，v1.3 的维护人员 Usage Analytics，以及 v1.4 的本机只读 Text-to-SQL prototype。RAG retrieval、rewrite、guard、rerank、generation profile、HTTP API 与 SQLite schema 均保持不变。

## 数据流与职责

```text
HTTP Assistant / Ask
  -> request_id + SQLite usage record
  -> route / safe trace / terminal status / timings
  -> positive or negative feedback
  -> private aggregate and review artifact
  -> user-confirmed failure classification
  -> versioned Eval candidate
```

SQLite 位于 `data_private/usage/fiscal_rag_usage.sqlite3`，保存不可重建的请求、回答、反馈与审核状态。Chroma 位于独立 index 目录，只保存由 corpus 派生的向量索引；不得查询、修改或复用 Chroma 内部数据库表来保存 usage 数据。

## 持久化边界

系统保存原始问题和回答、route、执行终态、脱敏 RAG trace、阶段耗时、稳定错误码、反馈和人工审核。系统不保存 Context/chunk 正文、prompt、embedding、provider payload、异常消息、堆栈、密钥、请求头、IP、User-Agent、cookie、姓名、账号或会话标识。

`completed` 只表示收到 SSE `done`，不表示回答正确。模型异常为 `failed`，客户端断开为 `aborted`，进程启动时遗留的 `started` 记录恢复为 `interrupted`。负面反馈只产生审核信号，不自动判定系统错误；未反馈也不作为中立评价。

原始记录默认保留 90 天。服务启动时清理一次，运行中每 24 小时最多再次清理一次。`manage_usage_db.py prune` 还会按 artifact 中的到期时间，成对删除默认目录中的 summary JSON/Markdown 和 Text-to-SQL query JSON/Markdown，并删除到期 review 和普通 backup；使用自定义输出路径时由操作者负责同等清理。人工晋升后的 Eval case 按版本化 Eval 资产管理；不含 production 流量且由 release verifier 固定引用的发布证据不属于普通 backup 自动清理范围。

## 流量分类

浏览器和普通 API 请求默认为 `production`。只有从 loopback 连接的 Assistant Eval runner 和性能 runner 才能分别标记为 `assistant_eval` 与 `performance_eval`；runner 使用非 localhost URL 时会直接拒绝运行，远程客户端伪造的分类头会按 `production` 处理。真实使用报告、反馈、人工审核和 Eval 候选只接受 `production`。该分类仍是内部统计标记，不是用户鉴权机制。

## 反馈接口

- `PUT /v1/assistant/feedback/{request_id}`：请求体 `{"rating":"positive"}` 或 `{"rating":"negative"}`，幂等新增或修改。
- `DELETE /v1/assistant/feedback/{request_id}`：撤销已有反馈。
- 只有 `production` 且 `completed` 的 Assistant stream 请求可以反馈；兼容 `/v1/ask`、Eval/性能流量、失败或中断请求不接受反馈。

前端只在收到 `done` 后显示评价按钮。评价不包含自由文本，避免额外收集可能包含敏感信息的说明。

## 分析与审核

生成私有使用报告：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_usage.py
```

命令默认覆盖数据库保留期内全部 production 记录，成对生成 `usage-summary-v2` JSON 和易读 Markdown。报告包含 endpoint、Assistant route、执行状态、错误码、反馈率、总耗时与各 route server total p50/p95、UTC 日趋势、RAG trace/source 完整性、review funnel、Eval-ready ID 和最多 20 条脱敏行动队列。行动队列只包含 request ID、route、状态、耗时、稳定错误字段、feedback 与 review signals。

默认报告不包含问题、回答、retrieval/rewrite query、约束、source 名、review reason 或 evidence。只有受控分析高频问题时才显式执行：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_usage.py --include-raw-questions
```

该选项只增加折叠空白后的精确问题及其次数/route，不增加回答或 trace，并把两个 artifact 标记为 `contains_raw_content=true`。报告支持带时区的 `--started-from/--started-to`、`--slow-ms` 和 `--queue-limit`；时间统一转为 UTC。

导出默认审核候选：负面反馈、未完成请求或总耗时不低于 6000 ms。

```powershell
.\.venv\Scripts\python.exe scripts\export_usage_review.py
```

可以重复使用 `--request-id` 只导出行动队列中的指定请求，也可用同一 UTC 时间范围过滤。任一 ID 未知、已确认或不再满足负面/非 completed/慢请求信号时，命令在创建 artifact 前整体失败。review JSONL 明确包含原始问题、回答和安全 trace，访问权限高于默认 Analytics 报告。

审核人员填写 `review_status=user_confirmed`、failure type、severity、expected route、answerability、reason 和 Eval 候选字段，再导入：

```powershell
.\.venv\Scripts\python.exe scripts\import_usage_review.py `
  --reviews-file <confirmed-review.jsonl>
```

failure type 固定为 `no_issue`、`routing`、`retrieval`、`knowledge_coverage`、`generation_support`、`generation_correctness`、`abstention`、`boundary`、`performance`、`upstream_transport` 或 `other`。

## Text-to-SQL 维护实验

维护人员可以在部署机本地提出一个结构化分析问题：

```powershell
.\.venv\Scripts\python.exe scripts\query_usage.py `
  --question "不同 route 的 completed Assistant 请求数量是多少？"
```

默认在 `data_private/usage/text_to_sql/` 以 exclusive-create 生成配对 JSON/Markdown。控制台只打印状态、结果行数和私有路径，不打印问题、SQL 或结果。artifact 保存维护问题、可审核 SQL、有限结果、schema/prompt/model 版本、耗时、截断标记、稳定错误码与 90 天到期时间，并标记 `data_scope=production_safe_views`、`contains_raw_request_content=false`、`contains_request_identifiers=true`。这些文件仍可能包含 request ID 和内部运营元数据，只允许维护人员访问。

模型只看到 `usage_requests`、`usage_timings`、`usage_feedback`、`usage_reviews` 和 `usage_source_stats` 的字段定义与指标语义，不看到真实样例行。五个连接内 TEMP VIEW 固定过滤 production，且不暴露 question、answer、retrieval/rewrite query、source 名、review reason、expected answer、evidence 或 topic。查询结果不会再次发送给模型。

执行器只打开已经存在的 SQLite 文件，不自动初始化；使用 `mode=ro`、`query_only`、authorizer、函数 allowlist、非递归单 SELECT 检查、执行超时与结果大小限制。默认最多 50 行，CLI 硬上限 100 行。它没有 raw mode，也不是鉴权后的通用 SQL 控制台。需要查看某个 request 的原文时，必须继续使用 `export_usage_review.py --request-id`，并遵循更严格的 ACL 与人工审核流程。

Text-to-SQL 适合临时 count、avg/min/max、列表和 JOIN。p50/p95、feedback 分母、行动队列和 review funnel 仍以 `summarize_usage.py` 的固定 Python 指标实现为权威口径。生成 SQL 和结果必须由维护人员复核，`completed` 不能解释为正确，缺失 timing 不能解释为 0，未反馈不能解释为中立。

## Eval 晋升

导出已确认且 `eval_candidate=true` 的记录：

```powershell
.\.venv\Scripts\python.exe scripts\export_eval_candidates.py
```

工具始终与冻结的 `assistant_eval_v1.jsonl` 比较，并拒绝重复 case ID、重复问题以及 category、expected route、answerability 相互矛盾的记录；`--existing-cases-file` 可重复提供其他版本数据集。`rag_answerable` 还必须补齐 expected answer、严格 relevant evidence 和 `retrieval_case_id`。输出是新的候选文件，不追加或覆盖冻结的 v1 数据集。

候选经人工确认后进入新的版本化 Assistant Eval 文件，再使用现有 `run_assistant_eval.py`、adjudication 和 summary/release gate。数据库中的 `source_request_id` 只用于私有 provenance。

## 维护与恢复

```powershell
.\.venv\Scripts\python.exe scripts\manage_usage_db.py prune
.\.venv\Scripts\python.exe scripts\manage_usage_db.py backup
```

`prune` 同时清理数据库、配对 Analytics 报告和默认生成的含原文 artifact。备份前也会按相同保留天数先清理数据库；备份使用 SQLite online backup API、以 exclusive-create 防止覆盖，失败时删除不完整目标。数据库及备份目录应仅允许服务账户和审核人员读取。恢复时停止服务、保留损坏文件、恢复备份，然后运行 `PRAGMA integrity_check`、启动服务并验证 feedback smoke test 及报告重建。
