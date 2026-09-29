# vLLM 基础用法（单卡 24GB RTX 4090）

这一目录演示最常用的 vLLM 工作流：离线批量推理、离线聊天，以及启动 OpenAI 兼容服务后进行普通和流式请求。

默认模型是 `Qwen/Qwen2.5-7B-Instruct`。首次运行时 vLLM 会从 Hugging Face 下载模型；若服务器通过 ModelScope 获取模型，先执行 `export VLLM_USE_MODELSCOPE=True`。模型需要访问权限时，请先在对应平台完成登录或配置令牌。

> 本目录假设服务器的 vLLM、CUDA 和 PyTorch 环境已经可用。运行客户端示例若提示缺少 `openai`，执行 `pip install -r requirements-client.txt`。

## 目录说明

| 文件 | 演示内容 |
| --- | --- |
| `offline_generate.py` | `LLM.generate`：一次对多个文本请求进行离线批量生成 |
| `offline_chat.py` | `LLM.chat`：传入 OpenAI 风格 messages 的离线聊天 |
| `serve.sh` | 用 `vllm serve` 启动 OpenAI 兼容 HTTP 服务 |
| `client_chat.py` | 用官方 OpenAI Python SDK 请求聊天接口 |
| `client_stream.py` | 用同一接口流式打印生成结果 |

## 1. 确认环境

```bash
nvidia-smi
python -c "import vllm; print(vllm.__version__)"
```

24GB 4090 建议从 `7B` 级指令模型开始。示例把 `gpu_memory_utilization` 设为 `0.85`，并将 `max_model_len` 限制在 `4096`，为驱动和运行时保留一些显存余量。若显存不足，优先降低 `--max-model-len`，其次再降低 `--gpu-memory-utilization`。

## 2. 离线推理（最短路径）

```bash
cd 01_vllm_basics
python offline_generate.py
python offline_chat.py
```

这里不会启动端口，适合批处理任务或熟悉 `LLM`、`SamplingParams` API。`temperature` 越低，输出通常越稳定；`max_tokens` 限制每个回答的最大新 token 数。

## 3. 启动 API 服务

```bash
cd 01_vllm_basics
chmod +x serve.sh
./serve.sh
```

另开一个终端请求服务：

```bash
cd 01_vllm_basics
python client_chat.py
python client_stream.py
```

服务默认监听 `0.0.0.0:8000`。若从另一台机器访问，把客户端地址替换为服务器 IP：

```bash
VLLM_BASE_URL=http://<服务器IP>:8000/v1 python client_chat.py
```

## 4. 常用替换项

```bash
# 换模型或端口（模型必须与显存相匹配）
MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8001 ./serve.sh

# 客户端使用相同的模型名和服务地址
MODEL=Qwen/Qwen2.5-7B-Instruct VLLM_BASE_URL=http://127.0.0.1:8001/v1 python client_chat.py
```

`serve.sh` 使用 `--generation-config vllm`，避免模型仓库中的 `generation_config.json` 悄然改变示例中未显式指定的采样默认值。生产环境请不要直接把未受网络或反向代理保护的服务端口暴露到公网；如需鉴权，可在 `vllm serve` 命令中添加 `--api-key <你的密钥>`，客户端对应设置 `VLLM_API_KEY`。

## 参考

- [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
