# 在同一组四张 H20 上顺序完成默认配置与新增配置对照

`run_pair.py` 先启动默认配置服务（baseline），完成两项精度与十二项性能工作负载，正常停止自己启动的服务，再启动新增配置服务（tuned）。两臂都使用固定提交 `eb34e115d041ac685fa38eadf8eae6c51d61dc23`、相同模型、匹配的原生扩展和完全相同的服务环境，只有目标 H20 配置 JSON 不同。baseline 必须实际删除包内 JSON；空 override 目录不能实现这一条件。

本目录独立于已冻结的 `review-20261028/validation`。`compare.py` 从该目录复制，保留原始命令构造和输出审计，再对本次任务的模式、并发、真实 Git 与原生生成文件做必要适配。正式入口只有 `run_pair.py`；不使用复制文件中旧的手动服务运行子命令。

## 运行前填写真实路径与分配

按 `manifest.example.json` 填写已核实的四张 GPU UUID、实际容器内路径、客户端版本、原生回执 SHA256 和模型快照来源。不要直接运行占位示例。baseline 与 tuned 的 `repo` 都必须有真实 Git 元数据，HEAD 为固定提交。服务 `python` 与包路径必须指向这两份真实源码。生成的扩展和包装文件可以来自同一份匹配构建，登记方法见下文。

模型和客户端可只读绑定。输出、数据集缓存、临时目录、编译缓存及日志放在独立数据盘；容器内路径必须与真实绑定一致。`storage.data_root` 是数据盘在容器中的根路径；数据集缓存也可以通过其他容器路径只读绑定，入口会核对它与数据盘根目录的真实设备号相同，不能只靠路径字符串。`storage.docker_root` 是 Docker 所在文件系统在容器中的可核实路径。入口会比较两者的文件系统设备号、容量和 inode；宿主机的 Docker 数据目录及实际挂载回执仍由 launcher 保存。不要把未挂载的 `/var/lib/docker` 路径写成假证明。

必须显式填写 `dataset_cache`（HF_HOME）、`hf_hub_cache`（HF_HUB_CACHE）、`hf_datasets_cache`（HF_DATASETS_CACHE）。已备官方缓存与 tokenizer 检查结果复用，不重新下载或遍历完整数据集。客户端和服务不继承操作终端环境，`PATH`、`HOME`、依赖 `PYTHONPATH`、库路径等都写入清单。服务的两份源码由各自 cwd 导入；填写 `client_source_repo` 和 `client_package_root`，客户端 `PYTHONPATH` 前置该固定源码，再加隔离客户端与公共依赖路径。`client_probe.py` 记录实际 `vllm.__file__`、`benchmarks.serve.__file__`、真实 Git HEAD 和受控差异，并核对导入路径；分布元数据版本与实际源码提交分别保存。两臂复用这一次固定客户端来源检查，保证 bench CLI 也来自本轮实现。不要让旧公共 vLLM 包抢在本轮源码之前。

两臂服务环境必须完全相同，显式 `VLLM_BATCH_INVARIANT=0`、相同四卡 `CUDA_VISIBLE_DEVICES`，禁止任何 `VLLM_TUNED_CONFIG_FOLDER`。入口检查真实 `/proc` 参数、cwd、环境、包位置和 `nvidia-smi` 选中卡 UUID；所有环境键必须声明且不得含凭证键。启动日志原始 bytes 会另存为不可变 `startup-log.snapshot.txt`，proof 只核对这个已捕获快照；完整 `server.log` 继续保留并监控增长，周期统计追加不会造成启动证明失配。日志还必须证明实际选择 Triton，以及 baseline 使用默认配置、tuned 加载目标 JSON 的绝对路径。参数中的 `--moe-backend triton` 本身不能代替实际选择日志。

主对照显式使用三步多 token 预测（MTP3），共享服务参数包含 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`。两臂的启动检查与正式评测使用同一服务进程和同一参数；不增加无 MTP 的另一套矩阵。

## 让资源池分配后再生成实际清单

`allocated_pair.py` 是正式资源池作业的薄入口。它在容器真正获得四卡后查询实际可见 H20 的 UUID 与驱动，核对本次 launcher 的 `OPERATOR_SWEEP_RUN_ID` 及原始存储回执，再构造清单。它不选择或预填宿主机 GPU。`--print-plan` 只输出计划并明确 GPU 核对尚未发生，适合在 CPU 上检查参数。

入口把只读 `/offline-cache` 中约 10 MiB 的官方缓存复制到本作业 `hf-cache`，保留 Hub 的相对 symlink，并显式设置 HF_HOME、HF_HUB_CACHE、HF_DATASETS_CACHE。datasets 的锁和派生缓存因此写入任务数据盘。权重不复制；使用只读 `/model` 和附带的既有结构回执，初始检查 62 个分片的文件长度与 stat，以及五份小元数据 SHA256；切换服务前后复查 stat，没有重读 328 GB 权重正文或重复已有分片头部核验。

Docker 的实际宿主机数据目录与设备号来自当前 launcher 原始回执，入口同时核对容器 `/work` 的设备号及当前可用容量/inode，不填假的容器 `/var/lib/docker` 路径，也不额外挂载 Docker 目录。预算为输出、缓存和日志合计 32 GiB，额外保留 20 GiB。服务与客户端只继承明确白名单中的 PATH/CUDA 动态库兼容变量，其他线程、离线、临时目录和编译缓存变量都显式声明，两个服务环境完全相同，OMP_NUM_THREADS、MKL_NUM_THREADS、OPENBLAS_NUM_THREADS 均固定为 1。

正式作业每臂在健康检查后先发一个真实模型请求，要求恰好完成一个输出 token，保存原始请求/响应，不计算精度分数。它与完整评测使用同一个 MTP3 服务进程。服务参数、模型及来源没有另行切换。

资源池完整对照的参数数组见 `pool-submit-argv.json`，只运行最终 tuned 服务启动与一个真实输出 token 的参数数组见 `pool-submit-smoke-argv.json`（`--smoke-only`）。两者使用不同 key 和输出目录，启动参数均为最终 MTP3 配置。该文件只是可审查命令，不会自动提交；实际 GPU 提交由主任务统一执行。入口所需 repo、native receipt、模型、客户端及缓存路径均为 CLI 参数。

## 原生构建回执

回执保留现有 `versions` 和 `files` 格式；`files` 列本轮实际使用的扩展和关键运行库绝对路径及 SHA256，`versions` 包含 `torch`、`cuda`、`triton`、`driver`、`python`、`build_receipt`。新增 `source_files` 对象登记组装到两臂源码中的生成文件，相对 repo 根目录填写，值为文件 SHA256：

```json
{
  "versions": {"torch": "真实版本", "cuda": "真实版本", "triton": "真实版本", "driver": "真实版本", "python": "真实版本", "build_receipt": "/data/实际构建回执.json"},
  "files": [{"path": "/data/实际native/_C.abi3.so", "sha256": "真实摘要"}],
  "source_files": {
    "vllm/_C.abi3.so": "相同文件摘要",
    "vllm/vllm_flashmla/flash_mla_interface.py": "实际生成包装摘要"
  }
}
```

入口逐项核对两臂生成文件。Git 差异检查只排除目标 JSON 和这份已核对列表；其余源码差异仍须与 `common_diff_sha256` 相符，额外未跟踪文件会失败。不会创建或伪造 `.git`。当前 DeepGEMM pin `0820030756945b2daec208e8e8255268360f337d` 通过 DeepJIT 的 DG 前缀读取 `DG_JIT_CACHE_DIR`，未设时再查 `DJ_JIT_CACHE_DIR`，默认才落到 `HOME/.dj`。示例显式把两者及 HOME、Torch extensions、ccache 等目录放在数据盘。每臂启动时检查来源；同一进程的工作负载间只复核进程身份，不重复全量扫描源码或权重。相同 resolved path 和文件 stat 的原生文件摘要在同一 runner 进程复用；新结果与变化日志不使用该复用。

## 先在 CPU 上核对客户端协议

`contract_probe.py` 使用实际 `lm_eval==0.4.13` 的 CLI 解析器、官方任务 YAML、真实 API 客户端构造器和请求构造函数。completion 客户端显式选择本地 Hugging Face tokenizer 后端；chat 客户端不使用 tokenizer。实际构造器源码已核对：只有 `tokenizer_backend=auto` 会做远端 tokenizer 能力探测，本入口显式选择后端，因此不进行该探测。全部缓存设为离线；不会请求模型服务、加载模型权重、加载完整数据集或生成分数。

在已备客户端和本轮源码的 CPU 容器里运行，示例路径按真实绑定替换：

```bash
python3 /evaluation/contract_probe.py --model /model --output /work/contract-probe.json
python3 -m unittest discover -s /evaluation/tests -v
```

probe 保存真实解析的参数、官方 YAML 摘要和请求体，并确认 CUDA 尚未初始化。它是新的 API 边界检查；旧的完整题量和 tokenizer 接口结果继续复用。

## 生成命令，再正式执行

下面命令只生成固定计划，不启动服务。`--output` 必须是新目录；若已经生成计划，正式执行时加 `--resume`。

```bash
python3 /evaluation/run_pair.py \
  --manifest /work/manifest.json --model /model \
  --native-receipt /work/native-receipt.json \
  --output /work/results/full-pair
```

核对 `plan.json` 后，用同样参数加 `--execute --resume`。服务默认最多等待健康检查 3600 秒，每项客户端命令最多运行 86400 秒；可用 `--health-timeout`、`--job-timeout` 改成双方共用的合理有限值。`--max-log-bytes` 默认每个客户端或服务原始日志限额 2 GiB；超额停止本脚本进程组并保留已有原件，不截断重写日志。宿主机 launcher 仍需预算、日志轮转和保留期限。不要同时向该 GPU 组或端口提交其他任务。

入口只终止自己创建的进程组，不结束已有服务或其他任务。已有健康服务占用目标端口时直接报错。服务的 PID、启动时刻、真实环境和启动/退出回执都保留；正常退出与失败都会执行自身服务清理。

## 固定测量内容与结果解释

| 任务 | 官方完整题量 | 示例数 | API | 生成上限 |
| --- | ---: | ---: | --- | ---: |
| GSM8K | 1319 | 5 | 原始 `local-completions`，不套 chat template | 256 |
| AIME25 | 30 | 0 | `local-chat-completions`，开启 chat template | 32768 |

两项精度任务并发 8，温度 0，采样种子和评测随机种子均为 20261028，保留官方 stop 字符串。GSM8K 官方 YAML 未指定生成上限，256 来自实际 0.4.13 API 基类默认；AIME25 的 32768 来自官方 YAML。服务和客户端都设置 40000 上下文长度，避免沿用历史 16384 限制；精度结果仍需检查原始逐题回答与完整样本数，不能声称已完成 API usage 的逐题 token 审计。

性能依次为输入/输出 token 数 8192/1、2048/512、512/1024、512/128，共 3 轮；每组 128 正式请求、16 预热请求，最大并发 128、到达速率 `inf`、随机长度比例 0、忽略 EOS。每轮客户端与模型采样种子相同，依次为 20261028、20261029、20261030。每个形状和轮次使用独立 `cache_salt`，两臂相同，避免前缀缓存跨工作负载复用。

`bench_record.py` 调用官方 `vllm bench serve`。它在官方完成 tokenizer 对齐之后、进入计时基准函数之前，保存实际请求 prompt 的逐条 SHA256 和预期长度，然后将同一批对象原样传给官方函数；不额外采样、不调用 RNG、不改变请求。两臂逐条 prompt 摘要必须相同。包装也记录真实 readiness、warmup、measured 请求的 success/error：readiness 重试单独登记，16 个 warmup 和 128 个正式请求必须全部成功；官方丢弃 warmup 返回值不能替代这一检查。每请求只做内存记录，计时结束后统一保存。原始 bench JSON 另外必须证明 128 个请求完成、0 个失败、128 条空错误以及全部输入/输出长度准确。此包装不保存额外一份大 prompt 明文，生成文本保留在官方 detailed JSON。

`pair-audit.json` 通过只说明完整题量、退出状态、实际输入一致和请求长度等检查通过，不自动说明精度非劣，也不把这组性能差异推广到其他长度或机器状态。两臂顺序固定，仍存在随时间变化的机器状态影响。

## 失败后恢复，不覆盖或挑选结果

保持同一清单、模型、回执、输出根目录和命令，加 `--resume --execute` 重启。入口重读每项已完成工作的原始输出并重新审计，合格才复用；已完成整臂不重新启动服务。未完成工作会写入 `attempt-002` 等新目录，首次失败的 stdout、stderr、argv、退出码、结果和服务证明保留原状。清单或逻辑命令变化时拒绝复用旧目录。若某项已经通过，入口不会重新执行它来挑更好的数据。

每个 attempt 保存 `command.json`、Linux 实际客户端环境、`stdout.txt`、`stderr.txt`、`exit.json`、服务证明引用以及原始 results/samples。服务单独保存在 `server-001` 等目录，包含完整 `server.log`、命令、进程、proof、显式启动确认与退出回执。`complete.json` 仅标记审计成功的 attempt；任何失败返回非零并保留现场。

原顺序编排的 15 项新增行为测试已在 H20-01 的 Linux CPU 容器全部实跑通过，真实 lm_eval 客户端协议检查也已通过。之后新增的分配入口补了 4 项本地行为检查（存储回执、带 symlink 的缓存复制、权重 stat 复查和启动请求）；只额外重跑受 Git 安全目录参数影响的那 1 项来源检查，五项均通过。原件、复用范围和失败修正见 [CPU 接入报告](../preparation-reports/evaluation-cpu-preparation.md)。旧 16 项工具测试和 3 项 CPU 配置回归仍复用。本轮尚未据此运行 GPU、产生精度/性能结论或修改上游 PR。
