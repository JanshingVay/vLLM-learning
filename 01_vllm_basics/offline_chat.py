"""离线 Chat 示例：消息格式与 OpenAI Chat Completions 类似。"""

from vllm import LLM, SamplingParams


MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    conversation = [
        {"role": "system", "content": "你是耐心的 vLLM 助教，回答要简洁准确。"},
        {"role": "user", "content": "PagedAttention 解决了什么问题？"},
    ]
    llm = LLM(
        model=MODEL,
        dtype="half",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    outputs = llm.chat(
        conversation,
        sampling_params=SamplingParams(temperature=0.2, max_tokens=256),
    )

    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
