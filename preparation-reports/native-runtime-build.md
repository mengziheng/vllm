# 为当前 PR 构建 H20 原生运行库

本轮已经按当前 PR 源码构建并安装本次模型路径需要的原生运行库，并在全新 CPU 进程中通过真实加载入口检查了文件来源和关键参数定义。运行包已冻结，下一步是共享 GPU 池的 TP4 模型启动和单请求检查。GPU 计算、完整模型质量和性能尚未由这份 CPU 回执验证。

此次修复的是当前 Python 包装与旧原生扩展之间的不一致。原生扩展是由 C++/CUDA 编译的 `.so` 文件。仅补一个能导入的 Python 文件，不能证明它调用的旧扩展接受相同参数，也不能证明扩展执行的是当前源码。

固定 PR 源码为 `eb34e115d041ac685fa38eadf8eae6c51d61dc23`。H20-01 上已经准备两份具有真实 `.git` 对象的独立源码副本；它们来自同一提交，Git 对象均包含在各自目录中。最终 baseline 只删除目标 H20 混合专家（MoE）层的配置 JSON，tuned 保留它；两边使用同一份本轮构建的原生文件。

## 为何需要补建这些目标

| 当前调用或依赖 | 实际发现 | 本轮处理 |
| --- | --- | --- |
| FlashMLA sparse decode | 当前包装传 14 个参数，公共 v032 扩展仅接受 13 个，缺少末尾 `enable_batch_invariant` | 按当前固定 FlashMLA 源码重建主扩展和 dense FP8 配套扩展，重新生成包装 |
| MLA 缓存写入 | 当前 `concat_and_cache_mla` 包装传 7 个参数，公共扩展仅接受 6 个，缺少 `skip_scale` | 重建当前主原生库 `_C_stable_libtorch` |
| Triton MoE 的排序、分块与求和辅助操作 | 参数数量相同，但当前相关 C++/CUDA 源码已变化 | 重建 `_moe_C_stable_libtorch`，不以参数数量相同代替源码相同 |
| GLM 的 FlashKDA prefill | 当前固定源码可复用；公共 wheel 缺少足够的目标级构建来源证明 | 使用同一固定源码重新构建 `_flashkda_C` |
| GLM 稀疏注意力的索引模块（sparse indexer） | 构造代码在 CUDA 平台明确要求矩阵计算库 DeepGEMM，当前 pin 与旧缓存不同 | 构建当前 `_deep_gemm_C`，安装配套 Python 包及运行时编译所需头文件 |
| CUDA attention 的第三代 FlashAttention（FA3）导入 | `fa_utils` 会导入 FlashAttention；当前 pin 已从旧 `9cd61de` 变为 `b273a026` | 只构建当前 FA3 目标；不构建本次路径不需要的 FA2/FA4 |

公共 v032 wheel 的原构建记录指向 `8b476fa8f60a5938daefd821b8bec5f49df36778`，且记录为 `working-directory`、`bundled_native_runtime=1`。它不能作为上述当前 PR 扩展的源码构建证明。原记录保存在 [common-v032-build.json](../native/common-v032-build.json)。本轮未修改公共目录、旧运行输入或原失败作业。

## 固定来源与构建范围

| 组件 | 本轮固定来源 |
| --- | --- |
| vLLM 主原生源码 | `eb34e115d041ac685fa38eadf8eae6c51d61dc23` |
| 主原生库 CUTLASS | `v4.7.1`，复用已有只读源码 |
| FlashMLA | `e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b` |
| FlashMLA CUTLASS | `4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b` |
| FlashKDA / CUTLASS | `17a037d98da546deb4591e967cf961a43c034d8b` / `5c149f52a436782210263fb2f19b354443a61c6a` |
| DeepGEMM / DeepJIT | `0820030756945b2daec208e8e8255268360f337d` / `3732a3b9a0a2e5126396908a362ba5abdf043b2f` |
| DeepGEMM CUTLASS | `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`，复用已有只读源码 |
| FlashAttention / CUTLASS | `b273a02621d3c94bc747df67204f591b67f38fae` / `62750a2b75c802660e4894434dc55e839f322277` |

构建使用当前仓库 CMake 的主原生目标段，再原样包含本次需要的外部目标配方。每一阶段复用同一 `build/core` 缓存，只增加缺失目标；没有构建无关的外部后端，也没有把这次选择目标的构建称为全部 main 目标的干净构建。阶段命令、CMake cache、实际编译命令、源码 pin、原始日志和产物 SHA256 分别保留。

运行环境固定为公共 CUDA 13 镜像：Torch `2.13.0+cu130`、CUDA toolkit `13.0.88`、GCC `13.4.0`、CMake `3.31.6`、Ninja `1.13.2`。没有升级 Torch 或安装另一个任务私有框架。当前主 CMake 对 Torch 2.13 的头文件修补只生成在任务构建目录；公共 Torch 文件没有改变。源码中主目标期望 Torch 2.14、FA 配方期望 Torch 2.4 的警告保留在原始日志，不能仅凭编译完成就外推全部模型行为。

## 存储与证据位置

宿主机任务根为 `/data03/tiger/vllm_pr58986_native_20261028_eb34e115`。Docker 数据目录 `/data00/docker` 在 `nvme0n1p1`，本任务数据在独立的 `nvme2n1p1`。启动前实测 data03 约有 196 GiB 可用空间、约 2.22 亿可用 inode；根目录及 containerd 所在文件系统约有 8.3 GiB 可用空间。

CPU launcher 实测挂载及写入探针确认 `sources`、`build`、`package`、`logs`、`tmp`、`cache` 均写入 data03。NVCC 的真实临时文件同样出现在该挂载的 `/work/tmp`。构建使用 64 GiB 输出预算，容器日志沿公共入口设置轮转；原始阶段日志和可复用构建产物保留到 PR 验证及评审结束，不自动删除。没有清理其他任务的数据。

## 当前验证范围

主原生库、MoE、FlashKDA、cumem allocator、spinloop、fs_io 的 125 个构建/链接步骤及安装已通过；当前 FlashMLA 两个目标的 64 个步骤及安装已通过；当前 DeepGEMM 主机扩展及配套包安装已通过；当前 FA3 的 195 个步骤及安装也已通过。所有阶段均在公共 CPU launcher 中执行，未挂载 GPU。

最终独立 CPU 作业 `sweep-unified-moe58986-8dd50e9bc7e6` 返回 `passed / exit0`。检查从 assembled 包优先路径启动全新 Python 子进程，实际导入了 FlashMLA 包装、DeepGEMM 首次使用时选择并导入计算库的入口、FA3 Python 接口及当前辅助算子。所有记录的 Python 文件和扩展均来自本轮 assembled 包；DeepGEMM 的正式入口选择了当前配套包，实际加载的 `_C_extension.abi3.so` 同样在该目录。内存映射中没有公共旧原生库，`torch.cuda.is_initialized()` 为 `False`。

新注册的 `sparse_decode_fwd` 已接受 14 个参数，末尾为 `enable_batch_invariant=False`；`concat_and_cache_mla` 已接受 7 个参数，末尾为 `skip_scale=False`。同时保留了 MoE、FlashKDA、FlashMLA 配套扩展、top-k 与 DeepGEMM 索引操作的真实参数定义。原始模块路径、加载路径、参数定义和内存映射见 [assembled-runtime-probe.json](../native/assembled-runtime-probe.json)，最终 launcher 原件见 [8dd50e9bc7e6.json](../evidence/sweep-unified-moe58986-8dd50e9bc7e6.json)。

本轮没有重新执行已完成的旧 CPU 参数解析回归或整仓检查。复用相同固定输入下的有效结果，只补新构建的导入、来源和参数定义；GPU 导入、实际计算、缓存语义、模型就绪、质量和端到端性能由随后真实模型作业验证。

## 冻结后的运行路径

宿主机只读输入为 `/data03/tiger/vllm_pr58986_native_20261028_eb34e115:/native-runtime`，模型作业的两臂工作目录分别是 `/native-runtime/assembled/baseline` 和 `/native-runtime/assembled/tuned`。两臂使用相同的 914 个已登记运行文件；baseline 唯一的 tracked 差异是目标 JSON 删除，排除该 JSON 和已登记运行文件后两臂 Git 差异都为空。

原生回执位于 `/native-runtime/receipts/native-receipt.json`，SHA256 为 `0ae2bd08b07775dbfcf3716adff67c43de387c72699a1480e6d87805952b7ff9`。该回执包含运行文件摘要、源码 pin、固定镜像及依赖、编译阶段回执路径；本地原件见 [native-receipt.json](../native/native-receipt.json)。冻结记录见 [FROZEN.json](../native/FROZEN.json)。正式模型作业只读绑定此根目录；此后不再修改它。

CPU 构建命名空间中的 `/work` 对应上述宿主机任务根，所以 CPU 原始日志中的 `/work/assembled/tuned` 在正式只读绑定后对应 `/native-runtime/assembled/tuned`。源码包的 Python 版本字符串为 `dev`，源码身份由真实 Git 提交与冻结回执表示。宿主机实际 NVIDIA 内核驱动为 `535.230.02`，保留现行固定镜像的 CUDA compatibility 条件，不能把 CPU 加载通过写成 CUDA 13 GPU 执行已经通过。
