"""CPU 教学模型：观察请求替换和按需分配 KV blocks，不预测 vLLM 性能。

简化设定：每个 prefill 消耗一个 step；每个 decode step 每请求生成一个 token；
没有 prefix cache、chunked prefill、抢占或真实 attention 计算。
"""

import argparse
import math


def simulate(continuous=True, block_size=4, capacity=16):
    waiting = [dict(name=name, prompt=prompt, output=output, generated=0, prefilled=False)
               for name, prompt, output in [("A", 5, 2), ("B", 5, 8), ("C", 9, 6), ("D", 3, 3)]]
    active = []
    step = 0
    while waiting or active:
        if continuous or not active:
            while waiting and len(active) < 2:
                active.append(waiting.pop(0))
        step += 1
        events = []
        peak = 0
        for request in active:
            if not request["prefilled"]:
                request["prefilled"] = True
                phase = "prefill"
            else:
                request["generated"] += 1
                phase = "decode"
            length = request["prompt"] + request["generated"]
            blocks = math.ceil(length / block_size)
            peak += blocks
            events.append(f"{request['name']}:{phase} len={length} blocks={blocks}")
        print(f"step {step:02d} | {'; '.join(events)} | KV blocks={peak}/{capacity}")
        if peak > capacity:
            print("容量不足：真实引擎可能延迟准入或抢占重算；教学模型在这里停止。")
            return
        completed = [r for r in active if r["generated"] >= r["output"]]
        for request in completed:
            print(f"        {request['name']} 完成，释放其 blocks")
        active = [r for r in active if r not in completed]
    print(f"共 {step} 个教学 step（不是毫秒，也不是 GPU 速度预测）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["continuous", "static"], default="continuous")
    args = parser.parse_args()
    simulate(continuous=args.mode == "continuous")
