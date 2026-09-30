# vLLM 核心项目：单卡推理服务性能实验室

你已经会启动模型。这个项目的目标是回答：**同样一张 4090，同样一个模型，为什么更改并发、输入长度、调度参数和 KV 容量后，速度和排队会发生变化？**

场景是一台多人共用的文档总结服务。短请求代表普通问答，长请求代表长文档，shared 请求代表所有人引用同一份长文档。用流式 HTTP 压测记录首段输出时间、完整响应时间、真实 token 数和引擎指标，最后形成可复查的实验报告。

推荐服务器版本是你已经使用过的 **vLLM 0.30.0**，模型仍然是 `Qwen/Qwen2.5-7B-Instruct`，单卡、FP16、4096 上下文。客户端和本地教学程序只用 Python 标准库，无需安装 OpenAI SDK 或额外压测依赖。客户端与模型服务分离，客户端不会再加载一份 GPU 权重。

## 项目地图

| 文件 | 读它时关注什么 |
| --- | --- |
| `serve.sh` | 调度预算、并发序列上限、KV 容量、缓存开关如何传给 vLLM |
| `benchmark.py` | 多个请求如何进入服务；怎样测 TTFT/TPOT；怎样解析 SSE 与 usage |
| `report.py` | 多轮实验结果并排比较 |
| `scheduler_model.py` | 本地逐 step 观察请求替换、KV blocks 增长与释放 |
| `kv_budget.py` | 从模型 GQA 配置计算每个 token 的 KV 字节数 |
| `test_lab.py` | 用本地 HTTP 测试服务验证采集代码，与 GPU 性能实验区分 |
| `results/` | 每次启动的配置与日志、每次压测的 JSON（Git 忽略） |
| `READING.md` | 从现象追到 vLLM 核心源码 |

```text
benchmark.py --concurrency N
  → N 个流式 HTTP 请求
  → vLLM waiting/running 队列与调度器
  → prefill / decode + KV block 管理
  → GPU 模型执行
  → SSE 文本 + usage
  → 客户端时间线、/metrics 快照、JSON 报告
```

## 先在本地学习，暂时不租 GPU

```bash
cd 02_core_lab
python scheduler_model.py --mode static
python scheduler_model.py --mode continuous
python kv_budget.py --tokens 2048 --cache-gib 1
python benchmark.py --workload mixed --describe
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v test_lab
```

观察 static 模式中 A 已经完成，但 C 要等 B 完成才能进入下一批；continuous 模式中 C 在下一 step 补进空位。教学模型模拟“请求完成后立即补位”的思想，**不是真实 vLLM 调度器、PagedAttention 实现或 GPU 性能模拟器**，没有计算 attention，也没有缓存复用/抢占/chunked prefill。

KV 估算默认采用 Qwen2.5-7B 的 28 层、4 个 KV heads、128 head dimension、每元素 2 字节：

```text
每 token 的 KV = 2(K 和 V) × 层数 × KV heads × head_dim × 元素字节数
               = 57,344 bytes = 56 KiB
2048 token 的一个请求 ≈ 112 MiB KV
1 GiB KV 理想上限 ≈ 9 个这样的请求
```

这里用 **KV heads = 4**，而不是 query heads = 28；这就是 GQA 减少 KV 占用的原因。按 block 向上取整，只是容量估算，尚未考虑共享前缀、元数据和其他开销。脚本的 16-token block 是教学假设，真实 block size 和实际容量要看启动日志。其他架构（MLA、混合模型、滑动窗口）不能直接照套公式。

## 服务器启动与第一次请求

先停止 `01_vllm_basics/serve.sh` 服务，一张 4090 一次启动一个模型服务。服务器先 `git pull`，再执行：

```bash
conda activate llm-infra
cd ~/autodl-tmp/project/vLLM-learning/02_core_lab
bash serve.sh baseline
```

默认走 ModelScope，复用已下载的权重。服务使用固定 API 模型别名 `core-lab`，所以即使从模型名改为本地权重路径，客户端也不必修改 model 字段：

```bash
MODEL=/你的模型绝对路径 bash serve.sh baseline
```

等 `Application startup complete` 后，第二个服务器终端运行：

```bash
conda activate llm-infra
cd ~/autodl-tmp/project/vLLM-learning/02_core_lab
python benchmark.py --requests 4 --concurrency 1 --label smoke
```

服务默认绑定 `127.0.0.1`，实验客户端在同一服务器运行。为解决你之前的 Clash 问题，本机 HTTP 请求主动绕过环境代理。服务端下载模型时仍保留你的环境设置；若出现 SOCKS 依赖报错，在服务端终端取消代理，或按已有方式补齐依赖。

## 实验 1：连续批处理与吞吐/延迟的取舍

保持 baseline 服务，依次运行：

```bash
python benchmark.py --requests 16 --concurrency 1 --label c1
python benchmark.py --requests 16 --concurrency 4 --label c4
python benchmark.py --requests 16 --concurrency 8 --label c8
python report.py results/c1-*.json results/c4-*.json results/c8-*.json
```

这是一种**闭环负载**：最多 N 个请求同时发送，完成一个才补下一个，客户端线程池里等待发送的时间不计入单请求 TTFT。它测的是这个客户端负载下的表现，不是固定到达速率下的生产 SLO。

先看整体输出 tok/s 是否上升，再看单请求 TTFT/TPOT 是否变差，不能拿一条回答的 tok/s 当成整个服务吞吐。完成短请求时空位被其他请求补上，是连续批处理的关键思想；本实验中的客户端并发本身不等于 GPU 每个 step 的真实 batch。

进一步对照：停止 baseline，运行 `bash serve.sh serial`，在并发 8 下压测。serial 只将 `max_num_seqs` 限为 1，**仍是 vLLM 引擎**，不是关闭连续批处理，也不是 Transformers 对照组。它用于观察服务器只允许一条 active sequence 时的排队与吞吐。

## 实验 2：拆开 prefill 与 decode

baseline 服务下运行：

```bash
python benchmark.py --workload short --output-tokens 128 --label short-in
python benchmark.py --workload long --output-tokens 128 --label long-in
python benchmark.py --workload short --output-tokens 512 --label long-out
```

输入长度看每条结果的 `usage.prompt_tokens`；`--repeats` 是重复文本次数，**不是 token 数**。输出强制 `ignore_eos=True` 到达 `max_tokens`，便于比较相同生成长度；这是性能实验负载，回答末尾可能不自然，不用于评价文本质量。

prefill 处理已知输入并建立 KV；decode 使用已有 KV 逐步生成。长输入通常使首段输出等待更久，长输出通常使 decode 阶段更久。客户端 TTFT 包含网络、服务排队和 prefill 等时间，不能仅凭它断言某一个阶段耗时。要结合 `/metrics` 中的 queue/prefill/decode histogram 计数与总和、服务日志及源码理解。

## 实验 3：共享长文档与前缀缓存 A/B

baseline 明确关闭 prefix caching；先在该服务下运行：

```bash
python benchmark.py --workload shared --requests 16 --concurrency 4 \
  --run-id prefix-ab-01 --label prefix-off
```

停止服务，再启动 `bash serve.sh prefix`。仍然使用同一个 run-id：

```bash
python benchmark.py --workload shared --requests 16 --concurrency 4 \
  --run-id prefix-ab-01 --label prefix-on
python report.py results/prefix-off-*.json results/prefix-on-*.json
```

两组使用相同 prompt。压测前均先发送一条不计入统计的共享前缀预热请求，再发送不同问题。on 组可以复用已算过的共享前缀 KV，off 组每次重新算。因为预热完成后才发送 measured 请求，避免把“第一批共同 miss”误当成缓存无效。

关注 `usage.prompt_tokens_details.cached_tokens`（如果版本提供）、原始指标中的 `prefix_cache_hits/queries` 计数变化和 TTFT。不同版本可能为计数器加 `_total` 后缀；完整 Prometheus 原文也写进 JSON。缓存影响的是重叠 token 前缀的 prefill，**不会缓存最终答案，也不会免除新 token 的 decode**。命中是 block/token 前缀级别；相似语义、文本中间相同不保证命中。

## 实验 4：chunked prefill 与调度 token 预算

baseline 的 `max_num_batched_tokens=2048`，wide-prefill 的值为 4096，其他主要配置相同，均启用 chunked prefill。两次分别启动并运行：

```bash
python benchmark.py --workload mixed --requests 16 --concurrency 8 \
  --run-id mixed-ab-01 --label budget-2048
# 重启为 bash serve.sh wide-prefill 后：
python benchmark.py --workload mixed --requests 16 --concurrency 8 \
  --run-id mixed-ab-01 --label budget-4096
python report.py results/budget-*.json
```

mixed 交替提交长、短 prompt。调度预算是一轮可处理的 token 数，并不等于并发请求数；`max_num_seqs` 才限制一轮序列数。chunked prefill 将长输入分块，使其能与正在 decode 的请求在预算内交错处理。

看吞吐、TTFT 和 TPOT 的变化，不预设哪个值必然更快。按请求 index 的奇偶分组再看长短请求差异；整体 p95 可能掩盖短请求的感受。若需更明显的长 prefill，可以增加 `--repeats`，但必须检查实际 prompt_tokens + output_tokens <= 4096；请求超长会报 HTTP 400，并在报告中计入失败。

## 实验 5：限制 KV 容量，观察等待与抢占

先用 baseline，再停止并启动 `bash serve.sh small-kv`，两次使用相同负载：

```bash
python benchmark.py --workload long --requests 32 --concurrency 16 \
  --output-tokens 256 --run-id kv-ab-01 --label normal-kv
# 重启为 small-kv 后：
python benchmark.py --workload long --requests 32 --concurrency 16 \
  --output-tokens 256 --run-id kv-ab-01 --label small-kv
```

small-kv 显式设置 **1 GiB KV pool**。vLLM 0.30 的 `kv_cache_memory_bytes` 指定缓存大小后，会覆盖通过 `gpu_memory_utilization` 推断 KV 大小的路径；不代表权重、激活、CUDA graphs 等其他 GPU 内存消失。

观察启动日志中实际 KV token 容量，压测时的 running/waiting、KV 使用率、queue time、preemptions 计数。容量不足可能先在准入阶段排队，不一定出现抢占；不出现抢占也有意义。`nvidia-smi` 看到的是已预留显存，`kv_cache_usage_perc` 表示 block 使用比例，两者不是同一个量。

需要修正一个常见误解：调小 `max_model_len` 限制单请求能有多长，能降低某些运行时开销和容量要求；**不保证已经按固定显存预算预分配的 KV pool 会按比例缩小**。本实验用明确的 KV 字节预算控制缓存容量。

## 如何读输出，避免测错

| 字段 | 含义 |
| --- | --- |
| `output_tokens_per_s` | 成功请求的真实 completion_tokens 总和 / 整批 wall time（含失败请求消耗的时间） |
| `ttft_p50_ms / ttft_p95_ms` | 从客户端实际发起请求到收到第一段非空文本；是 TTFT 的客户端近似 |
| `tpot_p50_ms` | (最后一段文本时间 - 第一段文本时间) / (真实输出 token 数 - 1)，是客户端平均 TPOT 近似 |
| `latency_p95_ms` | 从实际发起到收到流结束，含 usage 和结束事件 |
| `content_events_s` | 各段文本到达时间，可观察请求重叠，但一个 SSE chunk 可能含多个 token |
| `gauges` | 每 0.5s 的 running/waiting/KV 采样，短峰值可能漏采 |
| `metrics_delta` | 压测前后 counter、histogram sum/count 的差，含服务器可能延迟更新的影响 |
| `metrics_errors` | 指标采集失败原因；缺失指标显示 N/A，不用 0 冒充 |

失败会让命令返回非零退出码；不能删除失败行后再宣布吞吐提升。输入不是固定 token 数，比较之前检查 usage 分布。baseline 关闭前缀缓存，其他非 shared 负载在开头放唯一 ID 防止意外复用长前缀。一次只跑一组负载，服务上不要混入其他请求。

首次模型加载/编译不计入 benchmark；先做 warmup，再测请求。正式记录每组至少运行三次，注明冷启动、预热和缓存状态，保留服务 config/log；没有跨配置自动重启脚本，避免长时间误占租赁 GPU。关机前把 `results/` 下载回来，本地就能继续分析。

## 验收：你应能解释的五个问题

1. 为什么并发提高整体吞吐，却可能拖慢单个请求？证据是哪两列？
2. 为什么长输入影响首段输出，长输出影响持续生成时间？排队如何干扰判断？
3. 为什么共享文档只在 token 前缀匹配时复用 KV？哪里能看到实际命中？
4. 为什么大 token 预算与大并发上限不是一回事？混合长短请求如何影响体验？
5. GPU 已分配显存和正在使用的 KV blocks 有什么区别？KV 不够时为什么可能排队而不是 OOM？

把每组“配置 → 预测 → 实测 → 解释 → 下一步调整”写下来，这就是你的项目成果。可以在面试中具体说明单卡推理服务的调度、缓存和测量过程，而非只展示一个聊天接口。

参考：[vLLM 0.30 服务参数](https://docs.vllm.ai/en/v0.30.0/cli/serve/)、[性能调优](https://docs.vllm.ai/en/stable/configuration/optimization/)、[服务指标](https://docs.vllm.ai/en/v0.30.0/design/metrics/)、[Qwen 配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)。
