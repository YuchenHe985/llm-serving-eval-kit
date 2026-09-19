"""A tiny OpenAI-compatible streaming server with configurable timing, for tests and demos.

It is a model, not an inference engine: latencies are parameters and any number
measured against it describes the benchmark tool, never a GPU.
"""
from __future__ import annotations

import argparse
import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, ttft_s=0.05, tpot_s=0.01, capacity=4, fail_every=0, info=None,
                 omit_done=False, error_event=False):
        super().__init__(addr, _Handler)
        self.ttft_s, self.tpot_s, self.capacity = ttft_s, tpot_s, capacity
        self.fail_every = fail_every
        self.omit_done = omit_done      # end the stream cleanly but without "data: [DONE]"
        self.error_event = error_event  # emit an error event mid-stream
        self.info = info or {"version": "fake-0", "tp_size": 1}
        self._lock = threading.Lock()
        self._inflight = 0
        self.requests = 0

    def server_bind(self) -> None:
        # HTTPServer.server_bind() calls socket.getfqdn(), a reverse-DNS lookup that can block for
        # tens of seconds on machines without a resolvable hostname. Bind without it.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), port

    def handle_error(self, request, client_address) -> None:
        # Clients that hang up mid-response are normal in a load test; do not print a traceback.
        import sys
        if not isinstance(sys.exc_info()[1], (ConnectionError, BrokenPipeError)):
            super().handle_error(request, client_address)

    def start(self) -> "FakeServer":
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        srv: FakeServer = self.server  # type: ignore[assignment]
        if self.path == "/get_server_info":
            body = json.dumps(srv.info).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        srv: FakeServer = self.server  # type: ignore[assignment]
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        with srv._lock:
            srv.requests += 1
            count = srv.requests
            srv._inflight += 1
            inflight = srv._inflight
        try:
            if srv.fail_every and count % srv.fail_every == 0:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            factor = max(1.0, inflight / srv.capacity)
            tokens = int(req.get("max_tokens") or 16)
            time.sleep(srv.ttft_s * factor)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(s: str):
                b = s.encode()
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()

            for i in range(tokens):
                if srv.error_event and i == tokens // 2:
                    chunk("data: " + json.dumps({"error": {"message": "backend failed"}}) + "\n\n")
                chunk("data: " + json.dumps({"choices": [{"delta": {"content": f"t{i} "}}]}) + "\n\n")
                time.sleep(srv.tpot_s * factor)
            if (req.get("stream_options") or {}).get("include_usage"):
                chunk("data: " + json.dumps({"choices": [], "usage": {"completion_tokens": tokens}}) + "\n\n")
            if not srv.omit_done:
                chunk("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
        finally:
            with srv._lock:
                srv._inflight -= 1


def main() -> None:
    p = argparse.ArgumentParser(description="fake OpenAI-compatible streaming server (for demos and tests)")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--ttft", type=float, default=0.05)
    p.add_argument("--tpot", type=float, default=0.01)
    p.add_argument("--capacity", type=int, default=4)
    a = p.parse_args()
    s = FakeServer(("127.0.0.1", a.port), a.ttft, a.tpot, a.capacity)
    print(f"fake server on {s.url} (ttft={a.ttft}s tpot={a.tpot}s capacity={a.capacity})")
    s.serve_forever()


if __name__ == "__main__":
    main()
