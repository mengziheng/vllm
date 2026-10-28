# 正式对照入口已经通过 Linux CPU 接入检查

新增顺序编排的 15 项行为测试已在 H20-01 的公共 CPU 容器全部通过，包括真实 `/proc` 环境核对、服务启停、只清理自身进程组、失败现场保留、已完成结果复用、客户端错误导入路径拒绝、输入对象不变和预热失败拒绝。随后用真实 `lm_eval==0.4.13` 的 CLI、官方任务定义、API 客户端构造器及请求体构造方法核对了精度协议，两项都通过。整个准备过程没有运行模型、提交 GPU 工作、发送模型请求或生成分数。

最终成功的 launcher 回执为 `sweep-unified-moe58986-12d67483203d`，退出码 0。该轮仅补客户端协议，复用了前轮 `sweep-unified-moe58986-9d230e6763d4` 中已通过的 15 项新测试，先核对 `compare.py`、`run_pair.py`、`bench_record.py` 和测试源码与旧冻结副本相同，没有重跑它们。旧的 16 项 validation 工具测试、3 项 CPU 配置回归、官方离线数据完整加载及已有 tokenizer 检查继续复用。

## 本轮实际确认的精度参数

| 参数 | GSM8K | AIME25 |
| --- | --- | --- |
| 官方任务定义 | `tasks/gsm8k/gsm8k.yaml` | `tasks/aime/aime25.yaml` |
| API | `local-completions` | `local-chat-completions` |
| 题量计划 | 1319 | 30 |
| 示例数 | 5 | 0 |
| Chat template | 关闭 | 开启 |
| 并发 | 8 | 8 |
| 请求生成上限 | 256 token | 32768 token |
| 客户端上下文参数 | 40000 | 40000 |
| 温度 / 模型采样种子 | 0 / 20261028 | 0 / 20261028 |

GSM8K 的 256 来自实际 0.4.13 `TemplateAPI` 构造器默认值，官方 YAML 未覆盖；AIME25 的 32768 来自官方 YAML。两个请求体保留各自全部官方 stop 字符串。completion 客户端通过真实构造器从只读 `/model` 加载本地 tokenizer；chat 客户端显式关闭 tokenizer 后端。构造器源码已确认仅 `auto` 后端会探测远端 tokenizer 能力，本次显式后端不走该分支。输出确认 `cuda_initialized=false`、`requests_sent=0`、`scores_generated=false`。

这里核对的是客户端是否按协议构造请求，不代表完整题目已经生成提示或完成推理。完整 GSM8K/AIME25 的逐题回答与评分仍由正式 GPU 作业产生。

## 容器、存储与原件

实际任务根目录为 `/data03/tiger/vllm_pr58986_evalprep_20261028_eb34e115`，这轮 CPU 协议检查的冻结代码为其 `input/evaluation-v5`，绑定到容器内只读 `/evaluation`；正式运行的最终 v9 及窄改动验证记录见末节。`/client`、`/offline-cache` 和 `/model` 都只读绑定。公共镜像固定为 `sha256:45ec8bfe593206f8f157be556eb506958762716c98f6e1116be2225657586167`，Python 为 3.12.13，容器使用 `runc`、`network none`、空 `CUDA_VISIBLE_DEVICES`，没有 GPU DeviceRequests。

launcher 核对 Docker 数据目录 `/data00/docker` 设备号 66305，独立 `/data03` 任务设备号 66310；首次预检分别可用 120.23 GiB 与 193.33 GiB，本次预算 2 GiB，inode 充足。容器实测 `/work` 与 `/offline-cache` 都是设备号 66310。实际 mountinfo 证明 `/work`、`/tmp`、`/sweep-home`、各缓存和日志都落在独立任务数据盘；模型权重仍复用 `/data02` 只读副本。

- [Linux 15 项新测试原始 stderr](../evidence/evaluation-cpu/results-cpu-attempt003/new-tests/stderr.txt)、[退出码](../evidence/evaluation-cpu/results-cpu-attempt003/new-tests/exit.json)
- [真实客户端协议报告](../evidence/evaluation-cpu/results-cpu-attempt004/contract-probe.json)、[原始 stdout](../evidence/evaluation-cpu/results-cpu-attempt004/client-contract/stdout.txt)、[退出码](../evidence/evaluation-cpu/results-cpu-attempt004/client-contract/exit.json)
- [CPU 最终回执与实际 mountinfo](../evidence/evaluation-cpu/results-cpu-attempt004/cpu-receipt.json)、[launcher 成功回执](../evidence/evaluation-cpu/unified-environment/sweep-unified-moe58986-12d67483203d.json)、[launcher 原始日志](../evidence/evaluation-cpu/unified-environment/sweep-unified-moe58986-12d67483203d.log)
- [全部提交命令](../evidence/evaluation-cpu/launch-commands.json)、[原件文件清单与摘要](../evidence/evaluation-cpu/evidence-files.json)

## 保留的失败及修正范围

最初的提交 `49f9ebb4087a` 返回 75，原因是 native 构建持有 CPU 准备锁，没有创建容器。后续在构建正常结束后取得窗口，没有绕过资源锁或干预构建。

`cb6e1c62a77b` 和诊断轮 `412d5991c6e9` 在启动客户端后立即读取 `/proc/<pid>/environ` 得到空字节，即全部声明键缺失、没有任何新增键。诊断原件保留了 `{}` 和仅含键名的错误；没有暴露继承凭证。修正只对启动时的空环境做最长 1 秒、间隔 10 毫秒的重读。非空环境仍要求所有声明键和值完全相同。随后真实测试和协议进程均在第 2 次读取取得完整相等的环境，见 [捕获回执](../evidence/evaluation-cpu/results-cpu-attempt004/client-contract/environment-capture.json)。这证明本轮遇到的是启动期间的瞬态读取，不能据此推断其他环境中的非空差异也可忽略。

`9d230e6763d4` 的 15 项新测试全部通过；同轮客户端协议检查因按文件名递归搜索 `gsm8k.yaml` 同时命中标准任务与 llama3 变体而退出。该轮尚未构造 API 客户端或发送请求。随后只将任务来源改为 0.4.13 的明确标准路径，重新运行这项协议检查并通过；没有重跑已通过的 15 项测试。

## 正式运行仍需确认的内容

本轮没有验证 GPU 服务能使用最终匹配的 FlashMLA、DeepGEMM 和其余原生扩展启动；这些由独立构建与后续四卡作业完成。实际客户端 `vllm` 源码导入位置检查也在正式运行时执行：`client_probe.py` 分开记录 distribution 版本、`vllm.__file__`、`benchmarks.serve.__file__` 和真实 Git 提交，不允许相同版本字符串掩盖旧包导入。它要求清单明确填写 `client_source_repo` 与 `client_package_root`，推荐使用构建完成的 tuned 真实源码目录。

服务与性能协议保持固定：四张实际分配的 H20、TP4、Triton、`VLLM_BATCH_INVARIANT=0`，四种输入/输出长度、三轮、每轮 128 正式请求和 16 次预热。`bench_record.py` 在计时前记录对齐后的实际输入摘要，计时中只在内存登记 readiness、warmup 和 measured 的真实返回状态，计时后写原件。预热失败即拒绝该工作负载，readiness 重试独立记录；两臂每轮输入摘要必须相同。形状与轮次独立的 `cache_salt`、客户端种子和采样种子在两臂保持一致。

已只读核对 DeepGEMM pin `0820030756945b2daec208e8e8255268360f337d` 的 `csrc/runtime/jit.hpp` 与其固定 DeepJIT 的 `utils/env.hpp`、`cache/disk.hpp`：DG 前缀先读取 `DG_JIT_CACHE_DIR`，再查 `DJ_JIT_CACHE_DIR`，最后才默认 `HOME/.dj`。清单示例将这些缓存及 HOME、Torch extensions、ccache 等显式放在数据盘。原生回执校验仅发生在固定来源检查和每臂启动时，工作负载之间只核对进程身份，不重复哈希原生库；同进程相同真实路径与 stat 的摘要会复用。

## 最终资源池入口的补充检查

最终提交源已冻结在 `input/evaluation-v9`，完整参数数组为 [pool-submit-argv.json](../evaluation/pool-submit-argv.json)，该文件尚未由本准备任务执行。入口 `allocated_pair.py` 只在真实分配后查询四张 H20 UUID，核对当前 launcher 的原始存储回执，复制小型离线缓存并构造固定清单。模型回执为 [model-receipt-h20-01.json](../native/model-receipt-h20-01.json)，明确只复用既有分片结构核验；初始补查文件长度、stat 和小元数据摘要，切换服务前后复查 stat，没有重新读取 328,337,455,672 字节权重正文或分片头部。

主对照固定 MTP3，服务参数明确包含 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`。四个服务 worker 共用 16 CPU，OMP_NUM_THREADS、MKL_NUM_THREADS 和 OPENBLAS_NUM_THREADS 都固定为 1。完整作业每臂在健康检查后，用同一最终服务进程完成一个真实输出 token 的启动请求，随后自动执行正式精度与性能任务；不生成启动请求的精度分数。两臂先默认配置、后新增配置，任何命令或审计失败即停止，已完成工作只从原件重新审计后复用。

该入口新增四项本地 fixture 检查，分别覆盖当前 runid 的真实数据盘设备号、保留 Hub symlink 的缓存复制、只 stat 权重及检测文件变化、启动请求保留实际响应；另定向重跑了受 Git `safe.directory` 参数影响的一项来源检查，五项都通过，见 [定向输出](../evaluation/verification-allocated-local.txt)。这些 fixture 不构成 GPU 验证。

为避免完整服务日志的周期追加造成无关失配，启动时实际捕获的 bytes 原样保存为独立快照，proof 的摘要和配置/后端校验都指向该快照；完整 live 日志继续保留并监控增长。新增日志追加反例先失败、修正后通过，见 [快照定向测试](../evaluation/verification-startup-snapshot-local.txt)。没有为上述窄改动重跑旧的 15 项或 16 项套件及 3 项配置回归。[v9 同步回执](../evidence/evaluation-cpu/evaluation-v9-source-receipt.json) 记录冻结归档摘要及线程参数修正范围。
