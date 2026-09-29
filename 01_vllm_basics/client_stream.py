"""以流式方式打印 vLLM 服务生成的 token。"""

import os

from openai import OpenAI


BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=os.getenv("VLLM_API_KEY", "EMPTY"))
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "写一首关于 GPU 的四行小诗。"}],
        temperature=0.8,
        max_tokens=128,
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
