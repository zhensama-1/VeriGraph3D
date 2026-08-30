import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from verigraph3d.vlm import (
    BudgetedVLMClient, ChatCompletionsVLMClient, VLMBudget, VLMError,
    VLMRequest, VLMSettings,
)


class RetryHandler(BaseHTTPRequestHandler):
    calls = 0
    payloads = []

    def do_POST(self):
        type(self).calls += 1
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).payloads.append(json.loads(body))
        if type(self).calls == 1:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"temporary")
            return
        response = {
            "model": "mock-vlm", "choices": [{"message": {"content": "{\"ok\": true}"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass


def test_real_http_transport_retries_and_tracks_attempts():
    RetryHandler.calls, RetryHandler.payloads = 0, []
    server = ThreadingHTTPServer(("127.0.0.1", 0), RetryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        settings = VLMSettings(
            "chat", f"http://127.0.0.1:{server.server_port}/v1", "mock-vlm", "",
            max_retries=1, retry_delay_seconds=0,
        )
        client = ChatCompletionsVLMClient(settings)
        response = client.complete(VLMRequest("system", "health", json_schema={"type": "object"}))
        assert response.data == {"ok": True}
        assert client.usage.calls == 2
        assert client.usage.input_tokens == 12
        assert RetryHandler.payloads[-1]["response_format"]["type"] == "json_schema"
    finally:
        server.shutdown()
        thread.join()


def test_budget_blocks_calls_after_limit():
    settings = VLMSettings("chat", "http://127.0.0.1:1/v1", "unused", "", max_retries=0)
    raw = ChatCompletionsVLMClient(settings)
    raw.usage.calls = 1
    client = BudgetedVLMClient(raw, VLMBudget(max_calls=1))
    with pytest.raises(VLMError, match="budget exhausted"):
        client.complete(VLMRequest("system", "prompt"))
