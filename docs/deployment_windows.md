# Windows 公司内网部署

本文面向公司内网低频试用。服务不使用 Bearer Token，访问边界由公司网络隔离、Windows 防火墙和端口开放范围共同控制。服务使用单个 Uvicorn worker，并串行调用 Assistant/RAG workflow，以保护现有 rewrite/rerank trace 的一致性；最终回答通过同一 HTTP 服务的 SSE 响应逐步返回。当前只验证了 1 名实际测试者，不预设已验证的并发用户规模。

## As-built 部署记录

下表只记录用户或仓库已经确认的事实，不能用计划值代替实际值。

| 字段 | 当前记录 |
|---|---|
| 部署状态 | 已完成公司内网 Windows 单机部署 |
| 拓扑 | 单主机、单 FastAPI/Uvicorn 进程、单 worker、进程内请求锁；Web UI 与 API 同源 |
| 运行 profile | v1.1：Rewrite + Guard + Dense Top-20 + default reranker + Top-5 + DeepSeek `deepseek-v4-flash` |
| Python baseline | v1.1 冻结并已在当前 release 环境验证为 Python 3.13 |
| 数据/索引 | 私有 corpus 与 Persistent Chroma 独立迁移，Git 不承载私有数据 |
| 网络边界 | 公司内网；已通过另一台公司内网电脑使用内网 IP + TCP 8000 访问；API 无应用层鉴权 |
| 验证方式 | 1 名实际测试者完成跨电脑内网访问；另有 liveness/readiness、JSON/SSE smoke、索引 manifest、测试与 release verifier |
| 回滚 | 上一版代码、环境配置与成对 corpus/index 备份，恢复后重新执行健康检查和固定问题 |
| 实际部署日期范围 | 2026-08-25 至 2026-08-26 |
| 实际试用人数范围 | 1 名实际测试者；未验证并发用户数 |
| 性能采集 | 2026-08-26 在本机完成 48 个正式请求、3 组双请求 sanity check 和 1 次冷启动测量；详见 `evaluation.md` |
| 实际服务托管/启动方式 | 前台 PowerShell 手动运行 `.\.venv\Scripts\python.exe scripts\serve_api.py` |
| 后台常驻状态 | 尚未配置 Windows 服务、任务计划程序或其他后台托管 |
| 实际主机规格 | Intel Core i5-10400 @ 2.90 GHz；约 32 GB RAM（系统显示 31.8 GB） |

记录不包含主机名、IP、账号、公司名称或网段详情。

## 1. 准备代码、私有数据和 Python

使用公司批准的内部 Git、压缩包或文件传输方式复制项目。`data_private/` 不在 Git 中，必须单独迁移以下内容：

- `data_private/corpus/`：私有 Markdown 语料。
- `data_private/indexes/fiscal_guides_chroma_v1/`：Chroma 索引和 `manifest.json`。
- `data_private/usage/`：v1.2 SQLite 使用记录及受控备份；新部署可为空，升级或迁移时必须单独备份和恢复。

迁移前后应比较文件校验和，并保留一份只读备份。不要迁移 `.venv`，而是在目标电脑安装相同 Python 3.13 版本并创建新环境：

```powershell
py -3.13 -m venv .venv
$env:PIP_INDEX_URL = "https://<company-pypi-mirror>/simple"
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

公司 PyPI 镜像地址属于部署环境配置，不写入仓库。若镜像没有锁文件中的版本，应先由公司依赖管理流程同步对应 wheel，不能临时改用未经验证的版本。

## 2. 配置密钥和模型接口

从 `.env.example` 创建本机 `.env`，配置 embedding、rerank 和 chat 三类公司兼容接口。CCSwitch 只配置 Codex，不会自动向本 Python 进程提供这些变量。

```text
FISCAL_RAG_HOST=127.0.0.1
FISCAL_RAG_PORT=8000
FISCAL_RAG_LOG_LEVEL=INFO
FISCAL_RAG_USAGE_DB_PATH=data_private/usage/fiscal_rag_usage.sqlite3
FISCAL_RAG_USAGE_RETENTION_DAYS=90
```

`.env` 已被 Git 忽略。只允许运行服务的 Windows 账户读取该文件，例如由管理员按实际服务账户执行：

```powershell
icacls .env /inheritance:r /grant:r "<service-account>:(R)"
```

不要在命令历史、日志、截图、API 请求参数或 Swagger 示例中填写真实模型 Key。

使用数据库包含 90 天内的原始问题和回答。创建目录后，应由管理员按实际服务账户和审核人员设置 ACL；不要把数据库放进 Chroma index 目录：

```powershell
New-Item -ItemType Directory -Force data_private\usage | Out-Null
icacls data_private\usage /inheritance:r `
  /grant:r "<service-account>:(OI)(CI)(M)" `
  /grant:r "<reviewer-account>:(OI)(CI)(R)"
```

服务账户需要创建数据库、WAL/SHM 和备份文件的权限。实际 ACL 必须先在部署环境核对，不能把示例账户名原样执行。

## 3. 验证测试和索引

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\build_vector_index.py
.\.venv\Scripts\python.exe scripts\verify_release.py --allow-dirty
```

索引命令只验证并复用当前索引。出现 `Index already current` 才能继续；语料、embedding 模型或 chunk 参数确实变化时，停止服务后运行 `build_vector_index.py --rebuild`。该命令在 staging 中全量构建和验证，成功后切换，失败时保留最后成功索引。操作与恢复细节见 [Knowledge Base Maintenance](knowledge_lifecycle.md)。`--allow-dirty` 只用于部署预检；最终候选必须在干净工作区运行 `verify_release.py`，让它执行完整测试并生成私有 manifest。

## 4. 本机启动和检查

```powershell
.\.venv\Scripts\python.exe scripts\serve_api.py
```

另开 PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready

$json = @{ question = "单位基础信息怎么填写？" } | ConvertTo-Json
$body = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/ask `
  -ContentType "application/json; charset=utf-8" -Body $body
```

`/v1/ask` 是保留的完整 JSON RAG 兼容接口。验证流式 Assistant 事件可使用 Windows 自带的 `curl.exe`：

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/v1/assistant/stream `
  -H "Accept: text/event-stream" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary '{"question":"你好"}'
```

输出应依次包含 `start`、`route`、一个或多个 `delta`，最后为 `done`；RAG 问题还会包含不带正文的 `trace`。流建立后的模型错误通过 `error` 事件返回，此时 HTTP 状态已经是 200，应使用事件中的 request ID 排查。

`done` 的 `timings_ms` 应包含该 route 实际执行的阶段耗时。TTFT 必须以首个非空 `delta` 计算，不能使用即时 `start` 事件。

浏览器访问 `http://127.0.0.1:8000/` 可使用聊天界面；`http://127.0.0.1:8000/docs` 保留为 API 调试入口。聊天页面和 API 来自同一个 FastAPI 服务，不需要单独启动前端进程。若后续增加 IIS、Nginx 或其他内网反向代理，必须关闭该路径的响应缓冲并延长流式请求超时；当前响应已发送 `Cache-Control: no-cache` 和 `X-Accel-Buffering: no`。

首次启动会初始化或迁移 SQLite，并把上次异常退出遗留的 `started` 请求标记为 `interrupted`。启动失败时先检查数据库路径、ACL、磁盘空间和 `PRAGMA integrity_check`；不要删除数据库来绕过迁移错误。

完成一次 Assistant 请求后，用 SSE `done` 中的 request ID 验证反馈创建、修改与撤销：

```powershell
$requestId = "<completed-assistant-request-id>"
$feedback = @{ rating = "positive" } | ConvertTo-Json
Invoke-RestMethod -Method Put `
  -Uri "http://127.0.0.1:8000/v1/assistant/feedback/$requestId" `
  -ContentType "application/json" -Body $feedback
Invoke-RestMethod -Method Delete `
  -Uri "http://127.0.0.1:8000/v1/assistant/feedback/$requestId"
```

未知 request ID 应返回 `404`，未完成请求和 `/v1/ask` 应返回 `409`。不要使用真实敏感问题做部署 smoke。

使用同一个问题运行 `scripts/ask_rag.py`，核对 API 与 CLI 的 retrieval query、Guard 状态和 Top-5 来源。生成文本可能有轻微变化，但检索配置必须一致。

## 5. 开放公司内网访问

确认端口未占用后，在 `.env` 中显式设置：

```text
FISCAL_RAG_HOST=0.0.0.0
FISCAL_RAG_PORT=8000
```

`127.0.0.1` 只接受本机请求；`0.0.0.0` 表示 Uvicorn 监听这台电脑的所有网络接口，因此局域网内其他电脑才能连接。`0.0.0.0` 只是监听地址，客户端应使用服务电脑的实际公司内网 IP，例如聊天界面 `http://192.168.1.20:8000/` 或 Swagger `http://192.168.1.20:8000/docs`。

重启服务并用 `ipconfig` 确认主机内网 IPv4。由管理员创建只允许批准网段访问 TCP 8000 的入站规则，不能向任意网络开放：

```powershell
New-NetFirewallRule -DisplayName "Fiscal RAG API" -Direction Inbound `
  -Protocol TCP -LocalPort 8000 -RemoteAddress <approved-subnet> -Action Allow
```

从另一台公司电脑访问 `http://<internal-ip>:8000/` 即可使用聊天界面，也可以打开 `http://<internal-ip>:8000/docs` 调试 API，无需填写 Authorization。健康检查不会调用模型。当前接口本身不鉴权，因此防火墙规则必须只允许批准的公司网段访问，不得向公网或任意网络开放 8000 端口。

## 6. 使用数据库维护、日志与回滚

服务启动及运行中每 24 小时的首次业务请求会清理超过保留期的在线数据库记录。显式 `prune` 还会删除默认目录内已到期的 summary、review 和普通 backup；自定义 artifact 输出路径必须由操作者执行同等清理：

```powershell
.\.venv\Scripts\python.exe scripts\manage_usage_db.py prune
.\.venv\Scripts\python.exe scripts\manage_usage_db.py backup
```

维护人员应在部署机本地生成私有 Analytics，不通过无鉴权的 Assistant HTTP 服务暴露跨请求数据：

```powershell
.\.venv\Scripts\python.exe scripts\summarize_usage.py
```

需要进行不在固定报告中的临时结构化分析时，可使用独立的本机 Text-to-SQL prototype：

```powershell
.\.venv\Scripts\python.exe scripts\query_usage.py `
  --question "不同 route 的请求数量是多少？"
```

该命令不启动管理 API，也不经过普通 Assistant。它只查询 production 安全视图，默认最多返回 50 行，并将可审核 SQL 与结果写入 `data_private/usage/text_to_sql/` 的配对私有 artifact；控制台不显示问题、SQL 或结果行。该目录只允许维护人员读取，artifact 默认 90 天到期并由 `manage_usage_db.py prune` 成对清理。不要把该 CLI 包装成共享 Web 入口，也不要用它替代 `export_usage_review.py` 查看受控原文。

默认命令在 `data_private/usage/reports/` 以 exclusive-create 生成同名 JSON/Markdown，控制台只显示脱敏摘要、待审核数量和输出路径。两个文件均记录生成时间、UTC 统计范围、保留到期时间和 `contains_raw_content=false`。可使用带时区的 `--started-from/--started-to`、`--slow-ms`、`--top-n` 和 `--queue-limit` 调整范围；时间边界会统一转换为 UTC。

只有经批准分析原始高频问题时才使用 `--include-raw-questions`。它会令两个报告标记 `contains_raw_content=true`，且仍不包含回答、trace 或 source；这些文件必须与数据库使用相同 ACL 和 90 天保留策略。自定义 `--output-file/--markdown-file` 不由默认 prune 接管。

从默认脱敏行动队列选择需要查看原文的请求后，可按 ID 精确导出：

```powershell
.\.venv\Scripts\python.exe scripts\export_usage_review.py `
  --request-id <request-id> `
  --request-id <another-request-id>
```

也可以用相同的 UTC 时间范围批量导出。任一指定 ID 未知、已经人工确认或不满足负面反馈、未完成、慢请求条件时，命令整体失败且不留下部分 artifact。review JSONL 包含原始问题、回答和安全 trace，只允许审核人员访问；完成填写后依次运行 `import_usage_review.py`、重新生成 Analytics、再用 `export_eval_candidates.py` 导出人工确认的候选。

备份命令会先按同一保留天数清理数据库，再使用 SQLite online backup API，并拒绝覆盖已有文件；失败的备份不会留下不完整目标。不要在服务运行时直接复制 `.sqlite3`、`-wal`、`-shm` 三个文件，也不要只备份 Chroma；usage SQLite 与 Chroma 的职责和生命周期独立。

恢复演练步骤：停止服务，保留当前数据库作为故障证据，将受控备份恢复到配置路径，使用 SQLite CLI 或批准的数据库工具执行 `PRAGMA integrity_check`，再启动服务并完成健康检查、无敏感内容的 Assistant 请求和反馈 smoke。恢复后核对历史记录可查询、没有无法解释的 `started` 状态。数据库及含原文的 review/summary artifact 仍受 90 天保留和 ACL 约束。

服务只向 stdout 记录 request ID、路径、状态、耗时和异常类型，不记录问题、答案、Context、文档正文或 Key。用户报错时应提供响应头 `X-Request-ID`。

常见启动失败：

- 缺少环境变量：按错误中的变量名补齐 `.env`，不要打印变量值。
- 索引缺失或配置不兼容：先检查迁移完整性和 manifest；确认变化后在停服状态显式重建。只有 corpus 内容存在未发布变化时，服务会告警并继续最后成功索引。
- 远程电脑无法访问：依次检查监听地址、端口占用、Windows 防火墙、网段策略和主机 IP。
- 问答返回 `502`：使用 request ID 定位日志，再分别运行 embedding、rerank、rewrite smoke test 检查公司模型接口。

回滚时停止服务，恢复上一版代码、`.env` 和成对备份的 corpus/index。v1.1 不认识 v1.2 usage schema，因此保留 SQLite 备份但不要让旧服务修改它；以前台方式启动并重新执行健康检查及固定问题对照。

## 7. 正式部署后置项

试用稳定后再由公司运维完成固定 IP 或内网 DNS、低权限服务账户、Windows 服务托管、自动重启、日志轮转、监控告警和模型 Key 轮换。本阶段不引入 OAuth、SSO 或用户系统。

当前不使用 Docker/Kubernetes、Redis、任务队列、多 worker、独立前端服务或自动索引重建。网页静态资源由 FastAPI 同源提供。只有真实并发和失败数据证明有需要时，才重构为无状态并发 pipeline 或调整检索策略。

## 8. 部署机评估与发布

本机已分别执行冻结的 v1.1 验证和 2026-08-27 的 v1.2 验证。v1.2 selected Assistant Eval 为 54/54 completed；最终性能基线为 48/48 measured 和 6/6 concurrency completed；冷启动为 `6134 ms`。命令、失败 attempt、数据口径和实测数字统一见 [评估说明](evaluation.md)。raw artifact 留在 `data_private/`，性能记录不包含 question、answer 或 Context。

双请求结果用于量化单进程锁的 `queue_wait`，不是压力测试。只有该 baseline 和真实使用记录显示当前串行容量不满足需求时，才评估并发架构调整。

`v1.1.0` tag、Assistant Eval 和 2026-08-26 性能结果是冻结的历史证据。v1.2 已完成实际服务反馈 smoke、流量隔离、完整 Eval、性能复测、在线备份及独立恢复演练；最终库没有 `started` 记录，恢复副本 `integrity_check=ok`。前两轮性能运行暴露了真实上游连接失败，原始 artifact 必须保留。

v1.4 代码归档后应先运行一次明确使用 synthetic fixture 的 DeepSeek Text-to-SQL 评估，再在干净工作区运行不带 `--allow-dirty` 或 `--skip-tests` 的完整 verifier。该验证会生成私有 v1.4 manifest，并记录合成评估的 hash 和采用判断；其中 v1.3 的人工试运行门槛仍保持 pending，不能替代真实使用。

```powershell
.\.venv\Scripts\python.exe scripts\run_text_to_sql_eval.py
```

真实 usage 数据库 smoke 必须如实报告当前 production 行数。若为零，只能记录为空结果，不能用已有 Assistant Eval 或 performance Eval 行冒充 production。Text-to-SQL 未达到 8/10 answerable 与 2/2 refusal 门槛时保留 `not_adopted` 证据，不接入在线入口。

正式创建 `v1.3.0` tag 前，至少连续 5 个工作日生成 production 周期报告并检查完成率、上游错误、慢请求、负面反馈和遗留 `started`；至少一条真实行动队列请求完成人工审核；仅在 ground truth 充分时晋升 Eval；并完成数据库备份恢复与报告重建验证。若真实审核结论为 `no_issue` 或不适合 Eval，应保留该事实，不制造候选。v1.2 使用数据层继续作为已归档基础，不创建 `v1.2.0` tag。
