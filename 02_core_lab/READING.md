# 从实验现象进入 vLLM 核心

这个仓库使用安装好的 vLLM 提供推理。PagedAttention 的 kernel、调度器和 KV 管理在 vLLM 库内部。压测可以提供现象与指标，无法单独证明某个 kernel 的因果收益；下一步要结合原始实现阅读。

## 路线 1：请求如何进入、排队和被执行

从实验 1/4 的 running、waiting 和不同输出到达时间开始，阅读：

- [AsyncLLM](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/async_llm.py)：异步 generate 如何把请求交给引擎，并将输出返回给客户端。
- [Scheduler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py)：重点搜索 `schedule`、`token_budget`、`waiting`、`running`、`num_computed_tokens`、`allocate_slots`。
- [EngineCore](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py)：调度输出怎样成为一次模型执行，然后更新请求状态。

用纸画一轮调度：先有哪些 decode 请求？剩余 token_budget 能接纳多少 prefill？哪些请求仍在 waiting？不要假设“HTTP 一个请求 = GPU 一次完整 forward”。

## 路线 2：PagedAttention 与 KV 生命周期

结合 `scheduler_model.py` 观察的 blocks 增长/释放，再看真实实现：

- [KVCacheManager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/kv_cache_manager.py)：搜索 `get_computed_blocks`、`allocate_slots`、`free`。
- [BlockPool](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/block_pool.py)：搜索 free queue、ref_count 和 hash；注意“可复用”和“物理上已分配”不是同义词。
- [PagedAttention 原论文](https://arxiv.org/abs/2309.06180)：先读逻辑 block → 物理 block 的映射、按需分配、碎片和共享，再深入 CUDA 实现。

先回答：一个请求结束后，什么被释放？为什么逻辑连续的 token 不要求物理 KV 内存连续？cached blocks 怎样被复用和回收？注意论文设计与当下 V1 的具体实现并非每处完全相同。

本项目的 small-kv 对照测试的是容量变化，不是“关闭 PagedAttention”的对照；不要把容量实验结果全部归因于 PagedAttention。

## 路线 3：前缀缓存复用的到底是什么

先读 [自动前缀缓存设计](https://docs.vllm.ai/en/v0.30.0/design/prefix_caching/)，再跟进 BlockPool 的 block hash 和 Scheduler 取 computed blocks 的流程。运行实验 3 后，找到共享前缀的边界，解释为什么新问题仍需要新的 prefill 与 decode。

## 每次开 GPU 前的准备

本地先通读一条调用路径，写下预测，用 `--describe` 确认工作负载；开机只运行对应实验。保存 JSON 和 server config/log，关机后本地用 `report.py` 分析。每次只改变一个变量，遇到与预测不同的结果，再读对应源码。
