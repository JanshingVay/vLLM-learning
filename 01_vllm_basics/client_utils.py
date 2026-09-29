"""本机 vLLM 客户端共用的小工具。"""

import os
from urllib.parse import urlparse


def bypass_proxy_for_local_vllm(base_url: str) -> None:
    """避免全局 SOCKS/HTTP 代理阻断对本机 vLLM 的请求。

    仅当服务地址是 loopback 时清理当前 Python 进程的代理变量；访问远程
    vLLM 服务时会保留用户原有的网络配置。
    """
    hostname = urlparse(base_url).hostname
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        return

    for variable in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(variable, None)
