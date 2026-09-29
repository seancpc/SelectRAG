#!/usr/bin/env python3
"""決定性的假 OpenAI-compatible 串流伺服器(僅供測試與 gauntlet;不連任何真模型)。

行為:POST /v1/chat/completions(stream=true)→ 先送只含 role 的 chunk,再逐字送出
「<回覆:sha256(最後一則 user 內容)[:8]>」,末尾附 usage.completion_tokens,再送 [DONE]。
--drift N:第 N 題之後每則回覆多附一個字元(供一致性比對測試製造分歧)。
--cite:回覆末尾附「。[1]」,讓引用層與必要成分層有可判定的句子(預設關,
  否則回覆無 [n] → 無有效引用句 → 那兩層永遠拿到零列,斷言形同未執行)。
非串流請求走 judge 模式:回 JSON verdict(依主張雜湊),
主張含「<壞>」回非 JSON、含「<矛盾>」回 contradicted。
用法:python3 tools/fake_openai_server.py --port 0(印出實際 port)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


def make_handler(
    delay_s: float, drift_after: int, cite: bool = False
) -> type[BaseHTTPRequestHandler]:
    counter = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            body: dict[str, Any] = json.loads(self.rfile.read(n) or b"{}")
            if self.path != "/v1/chat/completions":
                self.send_response(400)
                self.end_headers()
                return
            user = [m for m in body.get("messages", []) if m.get("role") == "user"]
            seed_text = str(user[-1]["content"]) if user else ""
            if not body.get("stream"):
                self._judge(seed_text)
                return
            reply = "<回覆:" + hashlib.sha256(seed_text.encode()).hexdigest()[:8] + ">"
            counter["n"] += 1
            if drift_after and counter["n"] > drift_after:
                reply += "!"
            if cite:
                reply += "。[1]"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self._chunk({"choices": [{"delta": {"role": "assistant"}}]})
            for ch in reply:
                time.sleep(delay_s)
                self._chunk({"choices": [{"delta": {"content": ch}}]})
            self._chunk({"choices": [{"delta": {}}], "usage": {"completion_tokens": len(reply)}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _judge(self, seed_text: str) -> None:
            """非串流 + JSON:verdict 依主張雜湊;含「<壞>」回非 JSON(測解析失敗)。"""
            claim = seed_text.rsplit("待判定主張:", 1)[-1]
            if "<壞>" in claim:
                content = "not json"
            elif "<矛盾>" in claim:
                content = json.dumps({"verdict": "contradicted", "evidence_chunk": 1})
            else:
                h = int(hashlib.sha256(claim.encode()).hexdigest(), 16) % 3
                v = ["supported", "unsupported", "contradicted"][h]
                content = json.dumps({"verdict": v, "evidence_chunk": None if h == 1 else 1})
            payload = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": content}}]},
                ensure_ascii=False,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _chunk(self, obj: dict[str, Any]) -> None:
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


def serve(port: int, delay_s: float = 0.0, drift_after: int = 0, cite: bool = False) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), make_handler(delay_s, drift_after, cite))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--drift-after", type=int, default=0)
    ap.add_argument("--cite", action="store_true", help="回覆末尾附「。[1]」")
    args = ap.parse_args(argv)
    srv = serve(args.port, args.delay, args.drift_after, args.cite)
    print(f"PORT={srv.server_port}", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
