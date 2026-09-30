"""单卡、全层全注意力 GQA 模型的 KV 容量估算；无需 GPU。

默认参数来自 Qwen2.5-7B-Instruct config.json：28 layers, 4 KV heads,
head_dim = 3584 / 28 = 128。不能直接用于 MLA、滑动窗口或混合架构。
"""

import argparse
import math


def estimate(layers, kv_heads, head_dim, element_bytes, tokens, block_size, cache_gib):
    bytes_per_token = 2 * layers * kv_heads * head_dim * element_bytes
    blocks_per_request = math.ceil(tokens / block_size)
    bytes_per_block = bytes_per_token * block_size
    cache_blocks = math.floor(cache_gib * 2**30 / bytes_per_block)
    return dict(bytes_per_token=bytes_per_token, bytes_per_block=bytes_per_block,
                blocks_per_request=blocks_per_request,
                request_mib=blocks_per_request * bytes_per_block / 2**20,
                cache_blocks=cache_blocks, ideal_requests=cache_blocks // blocks_per_request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option, default in [("layers", 28), ("kv-heads", 4), ("head-dim", 128),
                            ("element-bytes", 2), ("tokens", 2048), ("block-size", 16)]:
        parser.add_argument("--" + option, type=int, default=default)
    parser.add_argument("--cache-gib", type=float, default=1)
    args = parser.parse_args()
    if any(value <= 0 for value in vars(args).values()):
        parser.error("所有参数必须 > 0")
    result = estimate(**vars(args))
    for name, value in result.items():
        print(f"{name}: {value}")
    print("理想上限不含权重、激活、CUDA graphs、元数据及共享前缀；实际容量以启动日志为准。")
