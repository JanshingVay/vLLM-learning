"""本地验证 HTTP/SSE、失败处理和报告；不依赖 GPU，也不伪造 GPU 性能结果。"""

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark import (make_prompt, parse_metrics, percentile, request_one,
                       run_benchmark, sse_events, summarize)
from kv_budget import estimate
from report import render


class FakeServer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'vllm:kv_cache_usage_perc{model_name="core-lab"} 0.25\n')

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if payload["prompt"] == "fail":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"prompt too long"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [dict(choices=[dict(text="", finish_reason=None)]),
                  dict(choices=[dict(text="你好", finish_reason=None)]),
                  dict(choices=[dict(text="世界", finish_reason="length")]),
                  dict(choices=[], usage=dict(prompt_tokens=24, completion_tokens=4))]
        for chunk in chunks:
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.flush()
        if payload["prompt"] != "truncated":
            self.wfile.write(b"data: [DONE]\n\n")


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, prompt="ok"):
        return request_one(self.url, "core-lab", "EMPTY", prompt, 4,
                           time.perf_counter(), 0, 5)

    def test_stream_ignores_empty_choices_and_bypasses_socks(self):
        with patch.dict(os.environ, {"ALL_PROXY": "socks5://127.0.0.1:1"}):
            result = self.request()
        self.assertIsNone(result["error"])
        self.assertEqual(result["preview"], "你好世界")
        self.assertEqual(result["usage"]["completion_tokens"], 4)
        # 两个内容事件含四个 token，不能拿事件数量代替 token 数。
        self.assertEqual(len(result["content_events_s"]), 2)
        self.assertAlmostEqual(result["tpot_s"],
                               (result["content_events_s"][-1] - result["content_events_s"][0]) / 3)

    def test_http_failure_and_truncated_stream(self):
        self.assertIn("HTTP 400", self.request("fail")["error"])
        self.assertIn("流中断", self.request("truncated")["error"])

    def test_sse_multiline_comments_and_eof(self):
        lines = [b": ping\r\n", b'data: {"a":\r\n', b"data: 1}\r\n", b"\r\n",
                 b"data: [DONE]\n"]
        self.assertEqual(list(sse_events(lines)), ['{"a":\n1}', "[DONE]"])

    def test_real_token_summary_and_failed_request(self):
        good, bad = self.request(), self.request("fail")
        stats = summarize([good, bad], 2)
        self.assertEqual(stats["success"], 1)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["output_tokens_per_s"], 2)
        self.assertEqual(percentile([1, 3], .5), 2)

    def test_metrics_labels_missing_and_histogram(self):
        data = parse_metrics('vllm:num_requests_running{model="a"} 2\n'
                             'vllm:num_requests_running{model="b"} 1\n'
                             'vllm:request_queue_time_seconds_sum 0.25\n'
                             'vllm:bad NaN\n# comment\n')
        self.assertEqual(data["vllm:num_requests_running"], 3)
        self.assertEqual(data["vllm:request_queue_time_seconds_sum"], .25)
        self.assertNotIn("vllm:bad", data)

    def test_shared_prefix_and_unique_prompt(self):
        shared = [make_prompt(i, "shared", 16, "run") for i in [0, 1]]
        self.assertEqual(shared[0].split("Task")[0], shared[1].split("Task")[0])
        self.assertNotEqual(make_prompt(0, "short", 16, "run"),
                            make_prompt(1, "short", 16, "run"))

    def test_kv_gqa_rounding(self):
        result = estimate(28, 4, 128, 2, 2048, 16, 1)
        self.assertEqual(result["bytes_per_token"], 57344)
        self.assertEqual(result["request_mib"], 112)
        self.assertEqual(result["ideal_requests"], 9)

    def test_end_to_end_benchmark_json_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "test.json")
            args = SimpleNamespace(base_url=self.url, model="core-lab", label="mock",
                                   workload="shared", requests=4, concurrency=2, repeats=4,
                                   output_tokens=4, timeout=5, run_id="test", output=filename,
                                   describe=False)
            self.assertEqual(run_benchmark(args), 0)
            result = json.loads(Path(filename).read_text())
            self.assertEqual(result["summary"]["success"], 4)
            self.assertEqual(result["summary"]["output_tokens"], 16)
            self.assertIn("mock", render([filename]))
            self.assertIn("25.00", render([filename]))
            self.assertTrue(result["metrics_before"])


if __name__ == "__main__":
    unittest.main()
