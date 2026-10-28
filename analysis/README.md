# 从两臂原始输出生成汇总

`summarize_pair.py` 读取 `evaluation/run_pair.py` 完成的一组 baseline/tuned 输出，生成 JSON 汇总。它不运行模型，也不重新检查模型权重或整份源码。只有两臂全部 14 项任务完成、成对审计通过、选定运行的原始输出重新核对一致时才会写入结果。缺少文件或检查失败时退出码为 1，并保留已有输出文件。

将 `analysis/` 与本次运行采用的 `evaluation/` 代码保持为同级目录，使用 Python 3 运行。GSM8K 双 filter 日志需要 v10 或后续的审计代码；不要让新汇总脚本导入仍按每题一行检查的旧版本。

```sh
python analysis/summarize_pair.py /path/to/finished-pair --output /path/to/summary.json
python -m unittest discover -s analysis -p 'test_summarize_pair.py'
```

准确率部分保留 lm-evaluation-harness 的指标名称、数值、标准误差和原始/有效样本数，并指向完整结果及逐题 JSONL。逐题输入核对按 `doc_id` 比较 `doc_hash`、`prompt_hash`、`target_hash`；某项哈希缺失时，分别对实际 `doc`、`arguments`、`target` 做稳定 JSON SHA256 摘要。`input_audit.sources` 明确记录每项采用的来源。

同一题在多个 filter 中可以有多行记录；filter 是对同一条生成回答采用的答案提取或匹配规则。例如 GSM8K 的 `strict-match` 与 `flexible-extract` 各保留 1319 行，共 2638 行，仍只计 1319 道独立题目。每个 filter 内不得重复题目，同一题跨 filter 的输入、reference 和原始生成回答必须一致；提取后的回答和评分可以不同。`sample_log` 分别记录每臂日志行数、独立题数和各 filter 数量。两臂的生成回答可以不同，不参与两臂输入相等判断。

吞吐量部分保留四种输入/输出长度的每次原值，并分别计算三次运行的中位数、最小值和最大值。`median_tuned_over_baseline_percent` 为 tuned 中位数除以 baseline 中位数后乘 100；`median_change_percent` 在此前减去 100。它们分别报告每秒请求数、每秒输出 token 数、每秒输入加输出 token 数；三次运行的范围不是置信区间。

测试只用明确标记的合成输入和手算数值检查汇总逻辑。完整成对 fixture 使用 GSM8K 双 filter 的 2638 行格式，以及 AIME25 的单 filter 30 行格式；另有单 filter 三行的独立解析测试，只检查分组行为，不代表完整 GSM8K 运行。测试通过不能说明模型准确率或性能已经通过验证；真实结果必须来自完整运行目录。


`audit_logged_docs.py` 另行核对最终两臂的测试文档是否与已保存的官方数据准备摘要相同。它直接读取四个已完成精度任务的原始逐题日志，要求每个 filter 完整覆盖题目、同一题跨 filter 的完整 `doc` 相等；随后按整数 `doc_id` 排序，对完整 `doc` 使用保留 Unicode、键排序、逗号/冒号分隔的 JSON，并在每题末尾包含换行，计算 SHA256。它不会删字段、修改字符串或重跑数据集/缓存检查。

```sh
python analysis/audit_logged_docs.py /path/to/finished-pair2 --output /path/to/logged-doc-audit.json
```

该命令默认读取同目录树下的 `evidence/offline-dataset-preparation/official-data-manifest.json`，也可显式传入 `--manifest`。只有四份完整测试文档的摘要都相同，JSON 的 `status` 才是 `matched`，退出码才为 0；摘要不同会记录实际/预期值并返回 1，不把日志文档逆变换成预期内容。缺少完整日志时不输出部分匹配结论。GSM8K train/few-shot 来源继续复用准备回执，不能从 test 日志重建。四项新合成测试只检查排序、Unicode、每题末尾换行、去重及额外字段保留；最终 pair2 已运行本项核对，四份测试文档摘要均与准备值匹配，见[实际审计报告](../reports/pair2-logged-doc-audit.json)。
