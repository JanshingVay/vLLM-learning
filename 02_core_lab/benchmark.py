"""标准库流式压测器：闭环并发、真实 usage、客户端时间线、Prometheus 快照。

阅读顺序：make_prompt -> request_one -> run_benchmark -> summarize。
线程只负责发送并发请求；GPU 调度、连续批处理和 KV 分配由 vLLM 完成。
"""

import argparse
import json
import math
import os
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting",
          "vllm:kv_cache_usage_perc")
LINE = "The document discusses GPU inference, memory bandwidth, token scheduling and efficient attention.\n"


def opener_for(url):
    # 本机服务不走 Clash；远程服务保留用户的代理设置。
    if urlparse(url).hostname in {"127.0.0.1", "localhost", "::1"}:
        return build_opener(ProxyHandler({}))
    return build_opener()


def make_prompt(index, workload, repeats, run_id):
    """repeats 是文本重复次数，不假装它等于 token 数；实际长度读取 usage。"""
    shared = workload == "shared"
    length = repeats if workload in {"long", "shared"} else 4
    if workload == "mixed":
        length = repeats if index % 2 == 0 else 4
    # 唯一标记放在开头，防止非 shared 负载意外复用长前缀。
    prefix = f"Document {run_id}-{'shared' if shared else index}:\n" + LINE * length
    return prefix + f"\nTask {index}: Continue explaining the document in detail.\nAnswer:"


def sse_events(response):
    """按空行分隔 SSE 事件；兼容注释、空 choices、[DONE] 和多行 data。"""
    data = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield "\n".join(data)


def request_one(base_url, model, key, prompt, output_tokens, origin, index, timeout):
    payload = dict(model=model, prompt=prompt, temperature=0, max_tokens=output_tokens,
                   ignore_eos=True, stream=True, stream_options={"include_usage": True})
    req = Request(base_url.rstrip("/") + "/completions",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    started = time.perf_counter()
    result = dict(index=index, started_s=started - origin, content_events_s=[],
                  usage=None, finish_reason=None, error=None)
    parts = []
    done = False
    try:
        with opener_for(base_url).open(req, timeout=timeout) as response:
            for event in sse_events(response):
                if event == "[DONE]":
                    done = True
                    break
                chunk = json.loads(event)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                if chunk.get("usage"):
                    result["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
                    if choice.get("text"):
                        result["content_events_s"].append(time.perf_counter() - origin)
                        parts.append(choice["text"])
            if not done or result["finish_reason"] is None:
                raise RuntimeError("流中断：未收到 [DONE] 或 finish_reason")
            if result["usage"] is None or not result["content_events_s"]:
                raise RuntimeError("缺少 usage 或文本；不能据此计算 token 吞吐")
    except HTTPError as exc:
        result["error"] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:1200]}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    result["ended_s"] = ended - origin
    result["latency_s"] = ended - started
    events = result["content_events_s"]
    result["ttft_s"] = events[0] - result["started_s"] if events else None
    count = (result["usage"] or {}).get("completion_tokens", 0)
    result["tpot_s"] = (events[-1] - events[0]) / (count - 1) if count > 1 and events else None
    result["preview"] = "".join(parts)[:200]
    return result


def metrics_url(base_url):
    parsed = urlparse(base_url)
    # 标准服务是 /v1；若部署在代理子路径下，需调整这里。
    return f"{parsed.scheme}://{parsed.netloc}/metrics"


def fetch_metrics(base_url, key):
    url = metrics_url(base_url)
    req = Request(url, headers={"Authorization": f"Bearer {key}"})
    with opener_for(url).open(req, timeout=3) as response:
        return response.read().decode()


def parse_metrics(raw):
    """单卡服务：按名字聚合不同 label 的样本，保留原文便于复查。"""
    values = {}
    for line in raw.splitlines():
        match = re.fullmatch(r'(vllm:[a-zA-Z0-9_:]+)(?:\{.*\})?\s+([^\s]+)(?:\s+\d+)?', line)
        if match:
            try:
                value = float(match[2])
                if math.isfinite(value):
                    values[match[1]] = values.get(match[1], 0) + value
            except ValueError:
                pass
    return values


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(records, elapsed):
    success = [r for r in records if r["error"] is None]
    tokens = sum(r["usage"]["completion_tokens"] for r in success)
    ttfts = [r["ttft_s"] * 1000 for r in success if r["ttft_s"] is not None]
    tpots = [r["tpot_s"] * 1000 for r in success if r["tpot_s"] is not None]
    cached = [(r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
              for r in success]
    return dict(success=len(success), failed=len(records) - len(success), wall_s=elapsed,
                output_tokens=tokens, output_tokens_per_s=tokens / elapsed,
                prompt_tokens=sum(r["usage"]["prompt_tokens"] for r in success),
                cached_prompt_tokens=(sum(cached) if cached and all(v is not None for v in cached)
                                      else None),
                requests_per_s=len(success) / elapsed,
                ttft_p50_ms=percentile(ttfts, .5), ttft_p95_ms=percentile(ttfts, .95),
                tpot_p50_ms=percentile(tpots, .5),
                latency_p95_ms=percentile([r["latency_s"] * 1000 for r in success], .95))


def run_benchmark(args):
    key = os.getenv("VLLM_API_KEY", "EMPTY")
    # 固定 run_id 用于跨配置 A/B。默认新 ID 防止上一轮已缓存的前缀污染下一轮。
    run_id = args.run_id or secrets.token_hex(8)
    prompts = [make_prompt(i, args.workload, args.repeats, run_id) for i in range(args.requests)]
    if args.describe:
        print(json.dumps(dict(run_id=run_id, args=vars(args),
                              prompt_characters=[len(p) for p in prompts],
                              example=prompts[0][:300]), ensure_ascii=False, indent=2))
        return 0
    origin = time.perf_counter()
    # 普通预热不污染 measured 前缀；shared 用同一长前缀预热，不计入统计。
    warm_prompt = (make_prompt(-1, "shared", args.repeats, run_id)
                   if args.workload == "shared" else "Warmup: explain inference briefly.")
    warmup = request_one(args.base_url, args.model, key, warm_prompt, 16, origin, -1, args.timeout)
    if warmup["error"]:
        raise RuntimeError("预热失败，请检查服务和上下文长度：" + warmup["error"])
    before_raw, after_raw = "", ""
    metric_errors, samples = [], []
    try:
        before_raw = fetch_metrics(args.base_url, key)
    except Exception as exc:
        metric_errors.append(str(exc))
    stop = threading.Event()
    origin = time.perf_counter()

    def monitor():
        while not stop.is_set():
            try:
                gauges = parse_metrics(fetch_metrics(args.base_url, key))
                samples.append(dict(at_s=time.perf_counter() - origin,
                                    values={name: gauges[name] for name in GAUGES if name in gauges}))
            except Exception as exc:
                if not metric_errors:
                    metric_errors.append(str(exc))
            stop.wait(.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    records = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(request_one, args.base_url, args.model, key, p,
                                   args.output_tokens, origin, i, args.timeout)
                       for i, p in enumerate(prompts)]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(f"request={record['index']:02d} latency={record['latency_s']:.3f}s "
                      f"error={record['error']}")
        elapsed = time.perf_counter() - origin
    finally:
        stop.set()
        thread.join(timeout=5)
    try:
        after_raw = fetch_metrics(args.base_url, key)
    except Exception as exc:
        metric_errors.append(str(exc))
    before, after = parse_metrics(before_raw), parse_metrics(after_raw)
    deltas = {name: value - before[name] for name, value in after.items()
              if name in before and name.endswith(("_total", "_count", "_sum", "_hits", "_queries"))
              and value >= before[name]}
    summary = summarize(records, elapsed)
    data = dict(created_at=datetime.now(timezone.utc).isoformat(), label=args.label,
                run_id=run_id, config=vars(args), summary=summary, warmup=warmup,
                requests=sorted(records, key=lambda r: r["index"]), gauges=samples,
                metrics_errors=metric_errors, metrics_delta=deltas,
                metrics_before=before_raw, metrics_after=after_raw)
    dest = Path(args.output or f"results/{args.label}-{time.time_ns()}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # 不覆盖旧实验。JSON 是运行结果，不包含模型权重。
    with dest.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if metric_errors:
        print("提示：/metrics 采集不完整，详见结果 metrics_errors；不影响请求时间统计。")
    print(f"结果: {dest}")
    return int(summary["failed"] > 0)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="core-lab")
    parser.add_argument("--label", default="experiment")
    parser.add_argument("--workload", choices=["short", "long", "mixed", "shared"], default="short")
    parser.add_argument("--requests", type=positive_int, default=16)
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--repeats", type=positive_int, default=128)
    parser.add_argument("--output-tokens", type=positive_int, default=128)
    parser.add_argument("--timeout", type=positive_int, default=300)
    parser.add_argument("--run-id", help="跨配置复用相同负载 ID；每次完整 A/B 使用新的 ID")
    parser.add_argument("--output")
    parser.add_argument("--describe", action="store_true", help="只预览负载，不访问服务器")
    args = parser.parse_args()
    try:
        return run_benchmark(args)
    except Exception as exc:
        parser.exit(1, f"实验未完成: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
