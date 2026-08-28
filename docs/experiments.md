# Retrieval 实验登记表

本文记录 v1.1 的权威 Retrieval 实验以及后续独立 prototype。不同实验的 schema、数据与指标不得混合；在线架构以 [系统架构](architecture.md) 为准。

## 固定口径

- 私有语料：29 份 Markdown，1000 chunks。
- Chunking：Markdown-aware `1000/100`。
- Embedding：Qwen 1024 维。
- 核心指标：Hit@1、Hit@3、Hit@5、MRR。
- Ground Truth：Evidence-Centric，必须由 chunk 正文覆盖标注 evidence，不能只匹配 source/section。
- V1：50 条开发集，用于方案选择。
- V2：15 条独立 holdout。
- V3：10 条同术语/不同业务对象诊断集。
- V4：16 条、覆盖 16 个 source 的广泛 holdout。

状态含义：`adopted` 为 v1.1 active profile；`baseline` 为冻结对照；`rejected` 为已有证据不支持采用；`superseded` 为曾采用但已被后续证据取代；`diagnostic` 仅用于解释特定失败模式。

## 登记表

| ID | 唯一变量 | Split / N | Hit@1 | Hit@3 | Hit@5 | MRR | 状态 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| E01 Global Dense | 无 reranker/rewrite/guard | V1 / 50 | 0.620000 | 0.880000 | 0.920000 | 0.750000 | `baseline` | 冻结最小对照 |
| E02 Default reranker | 增加 `qwen3-rerank` | V1 / 50 | 0.740000 | 0.960000 | 1.000000 | 0.853333 | `adopted` | candidate 内排序是主要改进点 |
| E03 Custom instruction | 仅替换 reranker instruction | V1 / 50 | 0.680000 | 0.960000 | 0.960000 | 0.816667 | `rejected` | 专用 instruction 净退化 |
| E04 Conservative rewrite | 仅改变 retrieval query | V1 / 50 | 0.780000 | 1.000000 | 1.000000 | 0.876667 | `adopted` | 缩小用户问法与文档表述差异 |
| E05 Rewrite Guard replay | 冻结 rewrite 上做接受/回退 | V1 / 50 | 0.780000 | 1.000000 | 1.000000 | 0.880000 | `adopted` | 防止约束遗漏，不以均值提升为唯一目标 |
| E06 Candidate Top-10 | Top-20 改为 Top-10 | V1 / 50 | 0.780000 | 1.000000 | 1.000000 | 0.880000 | `superseded` | V1 无回归，但 V2 暴露 Rank-11 cutoff |
| E07 Top-20 recovery | Top-10 改回 Top-20 | V2 / 15 | 0.800000 | 0.933333 | 0.933333 | 0.855556 | `adopted` | 修复 `v2-010`，最终 Context 仍为 Top-5 |
| E08 Top-20 no-regression | 同一 recovery profile | V4 / 16 | 0.687500 | 1.000000 | 1.000000 | 0.833333 | `adopted` | 与 V4 Top-10 逐题 rank 一致 |
| E09 Metadata context | metadata 拼入 reranker 文本 | V1 / 50 | 0.720000 | 1.000000 | 1.000000 | 0.840000 | `rejected` | Hit@1 与 MRR 下降 |
| E10 BM25/RRF Hybrid | Dense+BM25 经 RRF 后 rerank | V1 / 50 | 0.780000 | 0.980000 | 0.980000 | 0.870000 | `rejected` | 未超过当前配置且新增 Top-5 miss |
| E11 Ambiguity slice | 固定 profile，诊断同术语异对象 | V3 / 10 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | `diagnostic` | 10 条均 Rank 1；不外推到所有歧义 |

V1 Top-20 的当前汇总为 Hit@1 `0.780000`、Hit@3/5 `1.000000`、MRR `0.880000`。表中 E05/E06 分开保留 Guard replay 与 Top-10 历史阶段，避免把后续采用状态改写成当时的实验事实。

## 关键 Ablation 解释

### Reranker 有效，专用 instruction 无效

Global Dense 到 default reranker 的 Hit@1 从 `0.62` 提升到 `0.74`，Hit@5 从 `0.92` 提升到 `1.00`，说明正确 chunk 经常已在候选中但排序靠后。把 instruction 改成财政操作导向后，Hit@1 降到 `0.68`，并产生 Top-5 miss，因此保留默认 instruction。

### Rewrite 有效，Guard 负责约束保护

Conservative rewrite 将 V1 Hit@1 提升到 `0.78`。Guard frozen replay 的平均指标变化很小，但它的设计目标是当 rewrite 遗漏角色或高风险状态时回退原问题，避免静默退化；不能因平均提升小而删除，也不能把它包装成第二检索器。

### Metadata 文本前缀与 Hybrid 未提供净收益

将 source/section/subsection 作为普通文本拼入 reranker 输入，使 Hit@1 下降 `0.06`、MRR 下降 `0.04`。这不是 Metadata Filtering 实验。

BM25 Top-10 与 Dense Top-10 经 RRF 融合再 rerank，在 V1 引入新的 Top-5 miss。它说明 lexical candidate 可能把相近但不够具体的内容挤入有限候选池；现有证据不支持线上启用 Hybrid。

## Top-20 Recovery 与 Failure Analysis

`v2-010` 是已确认的 Top-10 candidate cutoff failure：

1. 语料中存在正确的账号解冻步骤，且没有明显 chunking 截断。
2. Rewrite 和 Guard 保留了核心业务对象。
3. 正确 evidence 位于 Dense Rank 11，未进入 Top-10。
4. Reranker 看不到该 chunk，因此无法恢复。
5. 当时 generation 正确 abstain，没有用相似的“指标解冻”流程代替答案。
6. Top-20 对照把正确 evidence 恢复到最终 Rank 1，并生成完整且未混入其他流程的回答。

该变化使 V2 Hit@1 从 `0.733333` 提升至 `0.800000`，Hit@3/5 从 `0.866667` 提升至 `0.933333`，MRR 从 `0.788889` 提升至 `0.855556`。V4 保持原结果。证据支持采用 Top-20，但不证明继续扩大候选池或重新启用 Hybrid 会更好。

## Generation 证据

V1 的 50 条 Generation 已由用户人工确认：49 条 `supported/correct`，1 条 `partially_supported/partially_correct`，0 条 `unsupported/incorrect`。该结论基于开发集和当时实际 Top-5 Context，不能写成跨语料泛化结论。

V4 的 16 条记录为 Assistant 初审，尚未升级为用户最终确认，不能与 V1 的确认口径合并。

## 私有 Artifact 登记

以下路径均位于被 Git 忽略的 `data_private/evals/results/`：

| 实验/结论 | 主要 artifact |
|---|---|
| Dense baseline | `global_dense_v1_details.jsonl`、`global_dense_v1_error_analysis.md` |
| Default reranker | `dense_rerank_v1_details.jsonl` |
| Instruction ablation | `dense_rerank_v1_instruction_ablation_analysis.md` |
| Rewrite | `dense_rerank_v1_query_rewrite_details.jsonl`、`dense_rerank_v1_query_rewrite_error_analysis.md` |
| Guard replay | `dense_rerank_v1_query_rewrite_guarded_replay_details.jsonl`、`dense_rerank_v1_query_rewrite_guarded_replay_analysis.md` |
| Top-10 history | `dense_rerank_v1_candidate_k10_guarded_replay_details.jsonl`、`dense_rerank_v1_candidate_k10_guarded_replay_analysis.md` |
| V2 Top-20 | `dense_rerank_v2_candidate_k20_guarded_replay_details.jsonl` |
| `v2-010` trace | `dense_rerank_v2_v2_010_candidate_trace_details.jsonl`、`v2_010_retrieval_failure_investigation.md` |
| V4 Top-20 | `dense_rerank_v4_general_candidate_k20_guarded_replay_details.jsonl` |
| Metadata context | `dense_rerank_v1_metadata_context_details.jsonl`、`dense_rerank_v1_metadata_context_error_analysis.md` |
| Hybrid | `hybrid_rerank_v1_guarded_details.jsonl`、`hybrid_rerank_v1_guarded_analysis.md` |
| v1.1 current snapshot | `v1_1_top20_release_snapshot.md` |

旧 `current_best_*` 和 `generation_eval_v1_current_top10_summary.md` 已显式标为 `superseded`，只保留历史证据，不再代表 active profile。

## 决策总结

`adopted`：Conservative Rewrite、Rewrite Guard、Dense Top-20、默认 `qwen3-rerank`、`page_content` reranker input、Top-5 Context。

`rejected`：财政操作导向 instruction、metadata-context reranking、BM25/RRF Hybrid。

尚无证据驱动进入主线：Metadata Filtering、Hierarchical Retrieval、chunking 参数搜索或继续扩大 candidate pool。新的 Retrieval 实验必须由多个真实失败形成的明确 slice 触发。

## X01 Text-to-SQL Usage Prototype

该实验不改变 Retrieval，也不称为 Agent。唯一目标是验证维护人员能否用自然语言查询现有 usage 结构化元数据，同时保持 production 隔离和只读安全边界。

| 项目 | 定义 |
|---|---|
| 输入 | 维护人员自然语言分析问题 |
| 模型上下文 | 五个安全视图 schema、枚举和 v1.3 指标语义；无真实样例行 |
| 模型输出 | 严格 `{"sql":"SELECT ..."}` 或 `unsupported_query` |
| 执行 | SQLite `mode=ro`、TEMP VIEW、`query_only`、authorizer、资源与行数限制 |
| 数据 | 程序生成 synthetic fixture；production/Eval/performance 混合以验证隔离 |
| 指标 | denotation accuracy、refusal accuracy，以及生成/校验/执行成功率 |
| 门槛 | answerable 至少 8/10，sensitive refusal 2/2，安全写入测试 100% 阻止 |
| 在线状态 | 不接入普通 Assistant；结果达到门槛也仅保留为本机维护 prototype |

版本化 case 位于 `tests/fixtures/text_to_sql_usage_eval_v1.jsonl`。私有运行结果位于 `data_private/evals/results/text_to_sql_usage_v1_<run-id>.jsonl` 和配对 summary；每条记录都必须标记 `contains_real_usage_data=false` 和 `data_origin=synthetic_usage_fixture_v1`。

2026-08-27 首轮真实 DeepSeek 结果为 answerable `9/10`、sensitive refusal `2/2`，达到预设门槛，状态登记为 `adopted prototype`。唯一错误为 `t2sql-010 source_coverage`：模型将 `has_retrieval_trace` 解释成有 source trace，并额外限定 Assistant endpoint；gold 通过 `usage_source_stats.source_count` 判断真实 source 记录覆盖，形成 `result_mismatch`。该结果说明 prototype 对常规聚合已具学习价值，但相近布尔语义仍会混淆，因此不升级为普通用户功能，也不能免除 SQL/result 人工复核。

未来公开 benchmark 应使用独立 case loader、schema adapter、数据库与 artifact，只复用严格模型输出协议、安全执行器和结果比较器；不得把 benchmark schema 或样例混入 usage prototype。
