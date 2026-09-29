"""请求已启动的 vLLM OpenAI 兼容服务。"""

import os

from openai import OpenAI


BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")


def main() -> None:
    # 未在 serve.sh 设置 --api-key 时，服务端不会校验该值；客户端仍需要传入非空字符串。
    client = OpenAI(base_url=BASE_URL, api_key=os.getenv("VLLM_API_KEY", "EMPTY"))
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个简洁的技术助手。"},
            {"role": "user", "content": "用一句话解释 KV Cache。"},
        ],
        temperature=0.2,
        max_tokens=128,
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
