"""无需第三方依赖，将多轮实验 JSON 对比为 Markdown 表格。"""

import argparse
import json
from pathlib import Path


def number(value):
    return "N/A" if value is None else f"{value:.2f}"


def render(paths):
    rows = ["| 实验 | 并发 | 成功/失败 | 输出 tok/s | TTFT p50/p95 ms | TPOT p50 ms | 峰值 KV% |",
            "| --- | ---: | --- | ---: | --- | ---: | ---: |"]
    server_rows = ["| 实验 | 峰值 running/waiting | cached prompt tokens | 服务端平均 queue ms | 前缀命中计数比 |",
                   "| --- | --- | ---: | ---: | ---: |"]
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        s = data["summary"]
        kv = [sample["values"]["vllm:kv_cache_usage_perc"] * 100 for sample in data["gauges"]
              if "vllm:kv_cache_usage_perc" in sample["values"]]
        label = data["label"].replace("|", "\\|")
        rows.append(f"| {label} | {data['config']['concurrency']} | {s['success']}/{s['failed']} | "
                    f"{number(s['output_tokens_per_s'])} | "
                    f"{number(s['ttft_p50_ms'])}/{number(s['ttft_p95_ms'])} | "
                    f"{number(s['tpot_p50_ms'])} | {number(max(kv) if kv else None)} |")
        def peak(name):
            values = [sample["values"][name] for sample in data["gauges"] if name in sample["values"]]
            return max(values) if values else None

        delta = data["metrics_delta"]
        queue_sum = delta.get("vllm:request_queue_time_seconds_sum")
        queue_count = delta.get("vllm:request_queue_time_seconds_count", 0)
        queue_ms = queue_sum / queue_count * 1000 if queue_sum is not None and queue_count else None
        hits = delta.get("vllm:prefix_cache_hits_total", delta.get("vllm:prefix_cache_hits"))
        queries = delta.get("vllm:prefix_cache_queries_total", delta.get("vllm:prefix_cache_queries"))
        ratio = hits / queries if hits is not None and queries else None
        server_rows.append(f"| {label} | {number(peak('vllm:num_requests_running'))}/"
                           f"{number(peak('vllm:num_requests_waiting'))} | "
                           f"{number(s.get('cached_prompt_tokens'))} | {number(queue_ms)} | {number(ratio)} |")
    return "\n".join(rows) + "\n\n" + "\n".join(server_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    print(render(args.results))
    print("\n失败不能忽略；仅比较相同模型、工作负载、输出长度和其他配置的实验。")
