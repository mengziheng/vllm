# 上游检查为什么没有继续运行

本次只读核对时，PR #58986 仍指向 `eb34e115d041ac685fa38eadf8eae6c51d61dc23`。ReadTheDocs #35462635 和 GitHub 的失败都停在贡献者授权检查；没有证据表明这次失败来自代码检查或文档渲染错误。文档能否完整构建仍未验证。

[ReadTheDocs 原始日志](https://app.readthedocs.org/api/v2/build/35462635.txt) 显示 checkout 成功，最后执行 `bash docs/pre_run_check.sh`，退出码为 1。脚本发现 GitHub `pre-run-check` 的结论为 failure，随即停止。日志只有六条准备命令，没有安装文档依赖、执行 MkDocs 或渲染页面。API v2 的命令记录与原始文本日志一致。

[GitHub 失败 job](https://github.com/vllm-project/vllm/actions/runs/40648625013/job/122683215016) 的实际检查要求三种标签之一：`verified`、`ready`、`ready-run-all-tests`；或者作者至少已有 4 个合并到该仓库的 PR。当前公开 API 重新查询仍为 0，当前标签为 `bug`、`documentation`、`performance`、`ci/build`、`glm`，不含所需标签。上游 `pre-commit` job 因这个依赖失败而 skipped，尚未检查改动。

同一提交最新新增的 workflow 是 [PR title #40679204613](https://github.com/vllm-project/vllm/actions/runs/40679204613)，创建于 2026-10-28 11:20:36 UTC，结论 skipped。它没有替代或修复 `pre-run-check`；固定提交中的标题 workflow 也要求 `ready` 标签。当前 DCO 与 Meta 检查成功，CodeRabbit 和标题检查为跳过；没有据此声称全部上游检查通过。

目前无需针对这些失败修改文档、放宽工作流权限或绕过检查。可以继续精简本次 PR 的实际差异，并在本地检查受影响文件；上游授权和后续 CI、文档构建由项目正常维护流程决定。本次没有重跑 CI、添加标签、评论或联系任何人。

原始证据均在本目录：`rtd-build-35462635-raw-log.txt`、两个版本的 RTD API 响应、当前 PR/check-runs/status/workflow-runs 响应、作者 merged 查询、GitHub job 日志/annotations，以及该固定提交的授权脚本与工作流。每次读取均有 `.command.json` 或 `.request.json`，记录请求、退出状态或 HTTP 状态及采集时间。`diagnosis.json` 提供简短机器可读索引。诊断工具的已连接 GitHub 会话不可用，因此使用官方 GitHub CLI/API 和 RTD 公开 API 完成；没有写入上游。
