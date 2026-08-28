# Knowledge Base Maintenance

本文说明维护人员如何把 `data_private/corpus/` 中的 Markdown 变化发布到在线 Persistent Chroma 索引。当前知识库规模较小，维护策略固定为可靠的全量 rebuild，不使用增量 indexing、watcher、queue 或独立 ingestion service。

## 发布模型

`data_private/corpus/` 是待发布的工作语料，`data_private/indexes/fiscal_guides_chroma_v1/` 是最后一次成功发布的索引。编辑工作语料不会动态修改正在运行的 Chroma collection；只有维护命令完整成功后，新知识才会生效。

每次构建都会递归读取 UTF-8、扩展名不区分大小写的 `.md` 文件。逐文件记录相对路径、SHA-256 和字节数：

- 新增文件会增加 source 和相应 chunks。
- 修改正文或文件名会改变 source manifest；mtime 变化本身不会触发更新。
- 删除文件会从新索引中删除其全部 chunks。
- 重命名等价于删除旧 source 并新增新 source。
- 空文件或不能产生 chunk 的文件会使发布失败。
- 非 Markdown 文件不会进入知识库。

## 标准维护流程

1. 在 `data_private/corpus/` 新增、修改、删除或重命名 Markdown。
2. 停止 `serve_api.py`。构建必须在没有服务进程持有 Chroma 文件的维护窗口执行。
3. 从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts\build_vector_index.py --rebuild
```

4. 只有看到 `Index rebuilt, validated, and activated` 才表示发布成功。输出同时列出 embedding model、dimension、Markdown 文件数、chunk 数和已执行的 validation 类别。
5. 重启服务并检查：

```powershell
.\.venv\Scripts\python.exe scripts\serve_api.py
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

语料没有变化时，可运行不带 `--rebuild` 的命令进行无成本 current-index 检查。它不会调用 document embedding API：

```powershell
.\.venv\Scripts\python.exe scripts\build_vector_index.py
```

出现 `Index already current` 表示 source manifest、配置、manifest 和 Chroma collection count 一致。检测到变化时命令只报告 stale，仍要求维护人员显式添加 `--rebuild`，避免意外产生 embedding 成本。

## 构建、验证与切换

rebuild 在正式索引的同一父目录创建唯一 staging 目录，完整执行 ingestion、chunking、document embedding 和 Chroma 写入。正式索引在以下检查全部通过前不会被修改：

- 每份 Markdown 至少覆盖一个 chunk；
- expected chunk ID、正文和 metadata 与 Chroma 存储完全一致；
- document、chunk、collection count 和 source coverage 一致；
- document embeddings 的维度一致；
- 固定非敏感 smoke query 的 embedding 维度匹配，并能完成一次 Chroma similarity query；
- 构建结束时重新计算的 corpus manifest 与构建开始时一致；
- staging manifest、collection 名、模型和 chunk 参数一致。

验证通过后命令关闭自身的 Chroma client，再通过同文件系统目录 rename 将旧索引暂存为 backup、激活 staging，并重新打开新索引检查。激活失败时会立即恢复旧目录。成功后才清理 backup。

retrieval smoke 只证明 embedding 和 Chroma 查询链路可用，不证明新增知识的答案质量。重大内容变化仍应按 [评估说明](evaluation.md) 运行相关 Retrieval 或 Assistant Eval。

## 失败与恢复

ingestion、embedding、验证或 corpus 漂移失败都只影响 staging；旧正式索引保持不变。此时可以直接重启服务，系统会记录不含 source 名和正文的 `corpus_has_unpublished_changes` warning，并继续使用最后一次成功索引。

目录切换使用 `.fiscal_guides_chroma_v1.backup` 作为可恢复状态：

- 正式目录缺失但 backup 完整时，下次维护命令先自动恢复旧索引。
- 正式目录与 backup 同时存在且正式索引完整时，下次命令完成遗留清理。
- 正式目录损坏而 backup 完整时，命令恢复 backup，并保留损坏目录供排查。
- backup 本身无法验证或恢复有歧义时，命令停止并报告保留路径，不继续覆盖数据。

不要手工删除 staging、backup 或 failed 目录，除非已经核实正式索引可打开且相关故障证据不再需要。服务启动只允许 corpus 内容处于未发布状态；embedding model、chunk profile、schema、collection 或 count 不一致仍会阻断启动。

## Release 边界

`scripts/verify_release.py` 是 v1.4 冻结代码与私有证据的发布 verifier，仍要求 29 documents、1000 chunks 和 1024 dimensions。它不是日常 Knowledge Base Maintenance 入口，也不会由 rebuild 自动运行。知识库内容变化需要进入新的正式版本时，应独立更新相应 release contract 和评测证据，而不是放宽现有 v1.4 gate。
