# 修正官方逐题日志审计并复用已完成的原始评分

v10 只修复两处评测工具行为，没有修改模型、固定源码、原生运行环境、官方任务、请求参数、分数或阈值。v9 和原始模型输出继续保留。

实际 `lm_eval==0.4.13` 在 evaluator.py 的 filter 外层循环中，为每道题、每个 filter 追加一条 logged sample；evaluation_tracker.py 只按任务写一个 JSONL 文件。GSM8K 的 strict-match 与 flexible-extract 因此共同产生 2638 行，每个 filter 各覆盖完整 1319 题。AIME25 使用默认 none filter，仍为 30 行。

新审计要求官方 filter 集合完全匹配，各组 doc_id 都是完整的 `0..count-1` 且不重复，并逐题比较不同 filter 的 doc、三个输入/目标摘要、arguments、target 和原始 resps。过滤后的回答和指标允许因 filter 不同而不同。返回值新增 `logged_sample_rows` 和 `filters` 计数。

恢复未完成工作的 `run_job` 现在先查找最早的 exit0、无超时原因的原始 attempt。在原命令、环境以及原/新服务证明一致的条件下，使用新审计重读原结果，不再次发送模型请求。它检查原始实际客户端环境、proof 摘要、已捕获的启动日志快照、native receipt 摘要，以及原/新 source/config/hardware/model/env 的一致性。验证通过后只写新的 complete receipt，并明确标记 re_audit 理由和两份服务证明。验证仍失败就停止，不改成重跑来选择另一份结果。

非零退出、启动失败或命令/环境不匹配的 attempt 不会被提升为完成。已有完成记录仍由原始结果重新核对后复用。新 allocation 的硬件 UUID 若与既有 manifest 不符，原有 fixed-input 检查仍会拒绝混配。

新测试：官方 schema 6 项、退出成功后的重审恢复 4 项均先失败后通过。仅重跑直接受恢复分支影响的 3 项原测试，全部通过，没有重跑旧 15/16 项套件、CPU 配置回归或模型。真实 v9 baseline GSM8K 的完整 2638 行原件已离线重审通过，官方两项得分和原始文件未改写。

资源池新 key 见 `pool-submit-argv.json`，输出目录和 pair_id 保持不变，以便在实际分配仍为同四卡时复用 v9 的 exit0 原始评分。该文件尚未由此准备任务执行；GPU 提交由主任务统一处理。
