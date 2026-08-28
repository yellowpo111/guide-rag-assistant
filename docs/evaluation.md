# 评估方法与 Release Gate

本文集中定义 v1.1 的 Retrieval、Assistant、groundedness/source trace 和性能评估。最终结果只在对应 artifact 已真实生成并满足人工确认口径后填写。

## 当前状态

| 评估 | 数据规模 | 状态 | 可报告结果 |
|---|---:|---|---|
| Retrieval V1/V2/V4 | 50 / 15 / 16 | 已完成 | Hit@k、MRR、逐题 Failure Analysis |
| V1 Generation | 50 | 用户已确认 | 49 supported/correct，1 partial，0 unsupported/incorrect |
| Assistant Eval v1 | 54 | 真实 HTTP/SSE 与用户逐条确认均已完成 | 自动 route/macro-F1/trace completion 均为 1.0；人工 gate passed，无 blocker |
| 部署性能 baseline | 16 cases，48 正式请求 | 2026-08-26 在实际部署机完成 | completion 1.0；TTFT p50/p95 2893/4459 ms；total p50/p95 3054/5377 ms |
| 双请求并发 sanity | 3 pairs / 6 requests | 2026-08-26 在实际部署机完成 | completion 1.0；queue wait p50/p95 2076/4584 ms |
| 冷启动至 ready | 1 次独立启动测量 | 2026-08-26 在实际部署机完成 | 4307 ms |

## Retrieval Evaluation

Ground Truth 使用 Evidence-Centric 标注。只有 retrieved chunk 正文实际覆盖目标 evidence 才算命中；source、section 或 subsection 相同但正文不包含证据时不算命中。

指标：

- Hit@1/3/5：目标 evidence 是否在前 k 个结果中出现。
- MRR：首个严格 evidence 命中 rank 的倒数均值。
- Coverage failure：语料本身是否缺少标注证据。
- Top-5 miss：正确 evidence 未进入 generation context。

当前 frozen profile 为 Rewrite + Guard + Dense Top-20 + default `qwen3-rerank` + Top-5：

| Split | 角色 | N | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---:|---:|---:|---:|---:|
| V1 | 开发集/方案选择 | 50 | 0.780000 | 1.000000 | 1.000000 | 0.880000 |
| V2 | 独立 holdout | 15 | 0.800000 | 0.933333 | 0.933333 | 0.855556 |
| V4 | 广泛独立 holdout | 16 | 0.687500 | 1.000000 | 1.000000 | 0.833333 |

严格 evidence 指标用于稳定比较策略，但不等同于用户任务成功率。同一流程可能在不同文档中有语义等价内容；反过来，Retrieval 命中也不保证 generation 完整使用证据，因此必须结合人工评估。

## Assistant Eval v1

私有版本化数据集位于 `data_private/evals/assistant_eval_v1.jsonl`，共 54 条：

| Category | N | 目的 |
|---|---:|---|
| `rag_answerable` | 31 | 原样引用 V2/V4 问题与 Ground Truth reference |
| `rag_unanswerable` | 6 | 知识库无答案或证据不足时是否适当 abstain |
| `chat` | 6 | 问候、致谢、身份和能力说明 |
| `out_of_scope` | 6 | 明确超出助手能力边界的请求 |
| `routing_boundary` | 5 | 混合问候+业务问题和不确定路由边界 |

Case schema：

```json
{
  "schema_version": "assistant-eval-v1",
  "case_id": "...",
  "category": "rag_answerable",
  "question": "...",
  "expected_route": "rag",
  "answerability": true,
  "retrieval_case_id": "v2-001"
}
```

所有案例必须通过真实 `POST /v1/assistant/stream`，不能直接调用 Router 或 pipeline。runner 消费实际 SSE，记录 route、event 序列、trace、answer、timing 和错误。

### 执行

先启动 v1.1 服务，再运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_assistant_eval.py `
  --base-url http://127.0.0.1:8000 `
  --run-id <run-id> `
  --attempt 1
```

本机尚未启动服务时，可让 runner 在同一进程中临时启动仅监听 localhost 的 Uvicorn，结束后自动关闭：

```powershell
.\.venv\Scripts\python.exe scripts\run_assistant_eval.py `
  --base-url http://127.0.0.1:8000 `
  --start-local-service `
  --run-id <run-id> `
  --attempt 1
```

部署机评估仍应连接实际常驻服务，不使用该选项。

输出使用 exclusive create；同名文件存在时直接失败，不覆盖历史记录。若首次执行失败，保留 attempt 1，并使用相同 run ID 与新的 attempt：

```powershell
.\.venv\Scripts\python.exe scripts\run_assistant_eval.py `
  --base-url http://127.0.0.1:8000 `
  --run-id <run-id> `
  --attempt 2 `
  --case-id <failed-case-id>
```

不要用 retry 覆盖或删除失败记录。将完整 attempt 1 与 retry subset 合并为最终选择集：

```powershell
.\.venv\Scripts\python.exe scripts\merge_assistant_eval_attempts.py `
  --details-file <attempt-1.jsonl> `
  --details-file <attempt-2.jsonl> `
  --output-file <selected.jsonl>
```

合并器要求所有文件使用相同 run ID，并按 case 选择最高 attempt，最终必须覆盖全部 54 条。原始 attempt 文件不变。

### 自动指标

- route accuracy。
- route macro-F1 与每 route F1。
- 3x3 confusion matrix。
- SSE completion rate 与 error count。
- RAG trace completion rate。
- 知识型问题被误路由为 chat/out-of-scope 的 critical failure 清单。

这些指标描述执行和路由，不替代回答质量人工判断。

### 人工 adjudication

从选定 details 生成不覆盖的 review template：

```powershell
.\.venv\Scripts\python.exe scripts\create_assistant_adjudications.py `
  --details-file <assistant-details.jsonl> `
  --output-file <assistant-adjudications.jsonl>
```

每条必须由用户审核并填写：

- `route_correct`
- `answer_support`
- `answer_correctness`
- `abstention`
- `boundary_compliance`
- `source_trace_quality`
- `reason`
- `review_status=user_confirmed`

最终汇总：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_assistant_eval.py `
  --details-file <assistant-details.jsonl> `
  --adjudications-file <assistant-adjudications.jsonl> `
  --output-file <assistant-final-summary.json>
```

只要有缺失 case、未知 case、pending/null 字段或非 `user_confirmed` 状态，summarizer 就拒绝生成最终结论。人工指标按 category 分开报告，不合并为一个模糊总分。

重大 unsupported claim、错误业务步骤、能力边界越界、知识型 RAG 问题误路由，以及自动执行错误都属于 release blocker。

### 当前自动结果

Run `20260826T160100Z` 的 attempt 1 完成 53/54；`assistant-v2-002` 在 route/trace 后出现空 generation stream。该 case 的 attempt 2 又在 preparation 阶段遇到一次上游 `APIConnectionError`，attempt 3 完成。三个原始 attempt 均保留，selected 文件对该 case 采用 attempt 3，其余采用 attempt 1。

| 指标 | Selected 结果 |
|---|---:|
| Cases / completed | 54 / 54 |
| SSE completion rate | 1.000000 |
| Route accuracy | 1.000000 |
| Route macro-F1 | 1.000000 |
| RAG trace completion rate | 1.000000 |
| Critical RAG route failures | 0 |

该结果证明本次 selected 执行的路由和传输完整性；前两次失败同时说明外部模型/网络仍存在偶发失败风险。用户随后逐条复核了全部 54 条回答与 trace，最终 artifact 为 `assistant_eval_v1_20260826T160100Z_user_confirmed.jsonl` 和 `assistant_eval_v1_20260826T160100Z_final.summary.json`。

### 最终人工结果

| Category | N | Answer support | Correctness | Abstention | Boundary | Source trace |
|---|---:|---|---|---|---|---|
| `rag_answerable` | 31 | 31 supported | 30 correct，1 partial | N/A | 31 compliant | 31 sufficient |
| `rag_unanswerable` | 6 | 5 supported，1 partial | 5 correct，1 partial | 6 appropriate | 6 compliant | 6 partial |
| `chat` | 6 | N/A | N/A | N/A | 6 compliant | N/A |
| `out_of_scope` | 6 | N/A | N/A | 6 appropriate | 6 compliant | N/A |
| `routing_boundary` | 5 | 5 supported | 5 correct | 2 appropriate，3 N/A | 5 compliant | 4 sufficient，1 partial |

两条非满分结果均保留了逐题理由：`assistant-v4-012` 正确回答岗位和操作，但漏掉完整菜单路径；`assistant-noanswer-002` 正确拒绝代查实时余额，但没有准确区分“助手无账户访问权”和“业务系统可手动更新余额”。两者均未形成错误业务步骤、unsupported 核心结论或不安全越界。

最终 summary 为 `total_confirmed: 54`、`release_gate: passed`、`release_blockers: []`。指标按场景报告，不把 N/A 项与不同任务类型压成一个模糊总分。

## Groundedness 与 Source Trace

RAG `trace` 应包含 retrieval query、rewrite 状态和 1-5 条连续 rank 的来源记录，每条包含 rerank score。人工审核判断 Top-5 ranked trace 是否足以支持回答中的主要 claim。

当前 UI 展示的是 source/section/subsection 级 Top-5 trace，不是逐句 inline citation。因此：

- Eval 可以判断来源有效性和 claim-to-source 支持。
- 不能声称系统提供逐句引用或精确 span attribution。
- 本阶段不为了指标新增 citation feature；限制需在报告中保留。

## Performance Evaluation

### Timing 定义

服务端 `done.timings_ms` 按实际 route 包含以下阶段子集：

| 字段 | 含义 |
|---|---|
| `queue_wait` | 等待单进程锁 |
| `router` | Assistant route 判定 |
| `rewrite` / `guard` | RAG query preparation |
| `query_embedding` | query embedding API |
| `vector_search` | Chroma similarity search |
| `dense_retrieval` | dense retrieval wrapper total |
| `rerank` | rerank API |
| `rag_preparation` | rewrite 到 prompt/context 准备总耗时 |
| `generation_ttft` | generation 开始到首个非空 delta |
| `server_ttft` | 服务收到请求到首个非空 delta，包含 queue/preparation |
| `generation` | 完整 generation 阶段 |
| `server_total` | service workflow 总耗时 |

客户端 `client_ttft_ms` 从发出请求到首个非空 `delta`；`client_total_ms` 到 SSE 结束。嵌套阶段不可直接相加，否则会重复计算。

### 固定 workload

私有 case 文件 `data_private/evals/performance_eval_v1.jsonl` 固定为 8 RAG、4 chat、4 out-of-scope。每题 1 次 warm-up、3 次正式采样，共 48 个 measured requests；warm-up 不进入 p50/p95。随后执行 3 组双 RAG 请求，共 6 个 concurrency requests，单独报告 queue wait。

在真实内网部署机运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_performance_eval.py `
  --base-url http://127.0.0.1:8000 `
  --environment-label <sanitized-label>
```

raw JSONL 不保存 question、answer 或 Context，只保存 release/profile、脱敏环境摘要、case ID、expected/actual route、状态、answer 字符数、客户端时间、服务端阶段时间与错误代码。summary 按 route 和 overall 输出 completion rate、TTFT/total p50/p95，并输出各服务端阶段 p50/p95。

冷启动至 ready 必须在目标端口没有服务时测量：

```powershell
.\.venv\Scripts\python.exe scripts\measure_startup_ready.py `
  --base-url http://127.0.0.1:8000
```

### 实际部署机结果

采集日期为 2026-08-26。机器为 Windows、Intel Core i5-10400 @ 2.90 GHz、约 32 GB RAM，运行冻结的 v1.1 profile。16 个 case 各先执行 1 次 warm-up，再执行 3 次正式采样；以下 percentile 只包含 48 个正式请求。

| Route | N | Completion | Client TTFT p50 / p95 (ms) | Client total p50 / p95 (ms) |
|---|---:|---:|---:|---:|
| RAG | 24 | 1.000000 | 3954 / 4930 | 4310 / 5568 |
| chat | 12 | 1.000000 | 1710 / 2525 | 1888 / 2672 |
| out-of-scope | 12 | 1.000000 | 846 / 1066 | 849 / 1067 |
| Overall | 48 | 1.000000 | 2893 / 4459 | 3054 / 5377 |

首次顺序请求成功率为 `1.000000`，SSE completion rate 为 `1.000000`。RAG 的主要服务端阶段如下；阶段存在嵌套，不能相加为总耗时。

| RAG stage | p50 / p95 (ms) |
|---|---:|
| Router | 960 / 1200 |
| Rewrite | 1025 / 1400 |
| Query embedding | 117 / 227 |
| Vector search | 9 / 10 |
| Rerank | 375 / 436 |
| RAG preparation | 1541 / 2051 |
| Generation TTFT | 1418 / 1859 |
| Generation total | 1723 / 3201 |
| Server total | 4294 / 5555 |

3 组双 RAG 请求共 6 个请求，全部完成；queue wait p50/p95 为 `2076/4584 ms`，最大值为 `4680 ms`。这只验证了同时两个请求时进程锁会造成可见排队，不是压力测试、吞吐 benchmark、并发用户容量证明或生产 SLO。

在 8000 端口无服务时独立测量 1 次，从启动 `scripts/serve_api.py` 到 `/health/ready` 首次返回 200 为 `4307 ms`。原始性能 JSONL、聚合 summary 和冷启动记录位于被 Git 忽略的 `data_private/evals/results/`；性能 raw artifact 不含 question、answer 或 Context。该基线受本机状态、网络和外部模型服务时延影响，不能外推为其他环境结果。

## Release Verification

预发布检查：

```powershell
.\.venv\Scripts\python.exe scripts\verify_release.py
```

它会执行 `pip check`、完整测试、固定配置校验、私有路径 Git 跟踪检查、索引 current 校验，并生成带 corpus/artifact hash 的私有 release manifest。开发中的预检可用 `--allow-dirty` 或 `--skip-tests`；正式候选不得使用这两个参数。

annotated tag `v1.1.0` 的历史发布条件为：干净工作区、完整 verifier 通过、54 条 Assistant Eval 用户确认且 release gate passed、部署机性能报告完成、部署事实补齐、文档数字与 artifact 一致。该 tag 与对应 artifact 已冻结，不会因 v1.2 的真实案例而追加或覆盖。

## v1.2 使用数据闭环与 Eval 晋升

v1.2 不改变冻结的 RAG profile。它在 API 编排边界记录真实请求、终态、安全 trace、耗时和二元反馈，再通过人工审核将有价值的失败案例晋升到新的版本化 Eval 数据集。完整数据边界和 CLI 见 [使用数据闭环](usage_data_loop.md)。

Assistant Eval runner 和性能 runner 通过 localhost 分别标记 `assistant_eval` 与 `performance_eval`；非 localhost Eval URL 会被 runner 拒绝，API 也不信任远程分类头。真实使用汇总、反馈、审核与候选导出只接受 `production`，因此 warm-up、回归和性能流量不会进入真实 route、反馈率或高频问题统计。

闭环验收顺序如下：

1. 从负面反馈、非 `completed` 或慢请求导出 pending review JSONL。
2. 人工确认 failure type、严重程度、期望 route、answerability 和原因。
3. 只将 `user_confirmed` 且 `eval_candidate=true` 的记录导出为候选。
4. `rag_answerable` 候选补齐 expected answer、严格 evidence 和 `retrieval_case_id`。
5. 人工合并到新的版本化 Assistant Eval 文件，再复用真实 HTTP/SSE runner、adjudication 与 release gate。

数据库中的生产反馈不直接构成质量结论：缺少反馈不是中立评价，负面反馈也不是自动失败标签。跨版本比较使用 `service_version/profile_id` 与人工确认的 failure type；小样本反馈率不作为 SLO。

### v1.2 实际部署机验证

2026-08-27 在同一台 Windows、Intel Core i5-10400、约 32 GB RAM 的部署机完成 v1.2 验证。RAG profile 与 v1.1 相同，新增部分只有 API 编排层的 usage SQLite 与 feedback。

Assistant Eval run `20260827T023500Z` 的 attempt 1 完成 51/54，前三条在 route 前遇到上游 `model_request_failed`；attempt 2 只重跑这 3 条并全部完成。selected artifact 覆盖 54 条，SSE completion、route accuracy、macro-F1 与 RAG trace completion 均为 `1.0`，error count 为 0。失败 attempt 原样保留，没有覆盖或删除。

性能采集保留了三次完整运行。前两轮分别因上游 route/preparation 连接失败达到 47/48 和 44/48 measured completion，均不能作为通过基线。第三轮 run `20260827T025300Z` 独立重跑完整 workload，48/48 measured 与 6/6 concurrency 请求全部完成：

| Route | N | Completion | Client TTFT p50 / p95 (ms) | Client total p50 / p95 (ms) |
|---|---:|---:|---:|---:|
| RAG | 24 | 1.000000 | 3706 / 4674 | 4017 / 5392 |
| chat | 12 | 1.000000 | 1709 / 2021 | 1894 / 2194 |
| out-of-scope | 12 | 1.000000 | 806 / 1050 | 820 / 1062 |
| Overall | 48 | 1.000000 | 2465 / 4472 | 2600 / 5171 |

第三轮双请求 queue wait p50/p95 为 `1820/4982 ms`，最大值为 `5358 ms`。一次独立冷启动至 ready 为 `6134 ms`。相较 v1.1 单次冷启动 `4307 ms`，本次更慢；只有一次样本，不能将差异完全归因于 SQLite。三轮中观察到的上游连接失败是实际稳定性风险，成功的第三轮不能抹去前两轮失败。

usage 数据库在最终采集后包含 269 条非生产记录：59 条 `assistant_eval`、210 条 `performance_eval`，其中 260 completed、9 failed，0 条 production、0 条 started。在线备份恢复到独立副本后包含完整 269 条记录，`PRAGMA integrity_check=ok`，反馈创建和撤销 smoke 通过。

这些结果证明当前实现可完成闭环采集、流量隔离和一次完整基线，不构成长时间 soak、并发容量证明或质量 SLO。真实内网试用仍需继续观察上游失败率和异常 `started` 记录。

## v1.3 Usage Analytics 与 Failure Operations

v1.3 不修改 SQLite schema、Assistant API、前端反馈或冻结的 RAG profile。它将现有 production usage 记录转换为 `usage-summary-v2` JSON 与配对 Markdown，供本机维护人员观察使用量、route/终态分布、耗时、反馈、RAG trace 完整性、审核漏斗和脱敏行动队列。默认报告不含问题、回答、trace、source 名或审核原因；只有显式使用 `--include-raw-questions` 才加入折叠空白后的精确问题统计。

指标口径如下：

- 所有 overview、latency、daily trend、feedback、review 和 Eval-ready 统计只使用 `traffic_kind=production`；Assistant Eval、性能 warm-up 和 measured 流量全部排除。
- feedback rate 的分母只包含可反馈的 `completed` production Assistant 请求；未反馈不算中立或负面。
- 慢请求默认定义为 `total_duration_ms >= 6000`，缺少 timing 不按 0 ms 处理；p50/p95 继续复用 `metric_summary` 的计算口径。
- 行动队列是负面反馈、非 `completed` 或慢请求的审核信号，不是自动 failure classification。已 `user_confirmed` 的请求退出队列，并进入 review funnel；只有人工确认且 `eval_candidate=true` 的记录进入 Eval-ready。
- daily trend 按 UTC 日期分桶，不补没有请求的日期。高频问题首版只做精确文本聚合，不做语义聚类。

自动验收覆盖空 production 数据库、流量隔离、三种 route、五种终态、缺失/部分 timing、阈值边界、UTC 时间范围、默认脱敏、显式原始问题、配对 artifact 的 exclusive-create/失败回滚/到期清理、request ID 精确审核选择，以及真实 HTTP/SSE 请求经过反馈、人工审核并导出 Eval candidate 的闭环。开发预检命令为：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify_release.py --allow-dirty
```

`verify_release.py` 验证 v1.3 代码与 `usage-summary-v2` 契约，但 manifest 中的 `operational_trial_gate` 在人工证据完成前保持 `pending_manual_evidence`。正式创建 `v1.3.0` tag 前还必须满足：至少连续 5 个工作日生成 production 周期报告；没有无法解释的 `started`；至少一条真实行动队列请求完成人工审核；只有 ground truth 充分的案例才晋升 Eval；完成数据库备份恢复与报告重建。零 production 数据库可以生成合法空报告，但不能代替这些试运行证据。

## v1.4 Text-to-SQL Usage Prototype

v1.4 是独立的维护人员实验，不接入 Assistant API、frontend、Router 或 RAG Workflow，也不修改五张 usage 表。模型只接收五个 production-only 安全视图的 schema、枚举与指标语义，不接收 fixture 行或真实 usage 内容；模型输出严格限制为单条 SQL 或 `unsupported_query`。执行结果不会发送回模型。

诊断集 `tests/fixtures/text_to_sql_usage_eval_v1.jsonl` 固定包含 12 条：10 条可回答的 count、avg/min/max、列表或 JOIN 问题，以及 2 条要求敏感原文的拒绝问题。runner 程序化创建明确标记为 `synthetic_usage_fixture_v1` 的 SQLite，混合 11 条 production 和额外的 Assistant Eval/performance Eval 记录，用于验证流量隔离。模型只看到 schema，不看到 fixture 行或 gold SQL。

正确性按候选 SQL 与人工 gold SQL 在同一 synthetic fixture 上的执行结果比较，alias 可不同，浮点数按固定精度归一化；聚合题可忽略行序，最近/最高类问题必须匹配顺序。记录的稳定失败类型包括 model transport/output、unsupported、SQL rejection/timeout/execution、result mismatch 和 truncation。该诊断不是公开 benchmark，也不能证明任意问题上的 Text-to-SQL 泛化能力。

```powershell
.\.venv\Scripts\python.exe scripts\run_text_to_sql_eval.py
```

最低采用门槛是 10 个可回答问题至少 8 个 denotation 正确、2 个敏感问题全部安全拒绝，并且原表、系统表、写入与资源滥用测试全部被阻止。即使未达到门槛，details JSONL 与 summary JSON 仍作为 `not_adopted` failure analysis 保存，不接入在线入口。当前真实 usage 数据库的 production 数量为零；真实库 smoke 只能证明返回空 production 结果，不能冒充业务成效。

2026-08-27 首轮真实 DeepSeek 合成运行结果为：generation success `1.0`、validation acceptance `0.833333`、answerable execution success `1.0`、denotation accuracy `0.9`（`9/10`）、refusal accuracy `1.0`（`2/2`），因此 prototype decision 为 `adopted`。唯一失败 `t2sql-010` 是 source coverage 语义混淆：候选 SQL 使用 `usage_requests.has_retrieval_trace` 并把 Assistant endpoint 限定为分母，gold 使用 `usage_source_stats.source_count` 统计全部 RAG usage，结果不一致。该失败保留为 `result_mismatch`，没有针对诊断题修改 prompt。实际数据库 smoke 生成合法 SQL 并返回 0 行；底层仍只有 59 条 `assistant_eval` 和 210 条 `performance_eval`，0 条 production。

v1.4 verifier 保留 v1.1/v1.2 的历史证据检查和 v1.3 的 `pending_manual_evidence` 运营门槛，同时增加 Text-to-SQL schema/prompt identity、只读安全 smoke、数据库 hash 不变和最新 synthetic evaluation artifact 校验。当前阶段不创建 tag。
