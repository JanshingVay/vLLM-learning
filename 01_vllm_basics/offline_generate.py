"""最小离线批量生成示例：不启动 HTTP 服务，直接调用 vLLM 的 Python API。"""

from vllm import LLM, SamplingParams


MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    # 一次传入多个 prompt 是 vLLM 连续批处理能力最直观的使用方式。
    prompts = [
        "用一句话解释什么是连续批处理（continuous batching）。",
        "用三点说明 vLLM 为什么适合部署大语言模型。",
    ]
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )

    # 24GB 4090 的保守起点。max_model_len 越大，KV Cache 占用越多显存。
    llm = LLM(
        model=MODEL,
        dtype="half",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    outputs = llm.generate(prompts, sampling_params)

    for index, output in enumerate(outputs, start=1):
        generated_text = output.outputs[0].text
        print(f"\n=== 回答 {index} ===")
        print(generated_text)


if __name__ == "__main__":
    main()
