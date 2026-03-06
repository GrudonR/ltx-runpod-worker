"""Thin HTTP wrapper around the LTX handler for RunPod pod mode.

Runs on port 8000 (configurable via HTTP_PORT env var).
Exposes:
  POST /run      — submit a job (same payload as serverless input)
  GET  /health   — health check
  GET  /status   — returns whether model is loaded

On startup, pre-downloads model assets so first request is fast.
"""

import json
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import the handler logic from the main handler module
from handler import handler, ensure_assets, LOGGER

HTTP_PORT = int(os.getenv("HTTP_PORT", "8000"))
_ready = False
_loading = True


class LTXHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"ok": True, "ready": _ready, "loading": _loading})
        elif self.path == "/status":
            self.send_json(200, {"ready": _ready, "loading": _loading})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/run":
            self.send_json(404, {"error": "not found"})
            return

        if not _ready:
            self.send_json(503, {"error": "model still loading", "loading": _loading})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError) as e:
            self.send_json(400, {"error": f"invalid JSON: {e}"})
            return

        # The handler expects {"input": {...}} format from serverless,
        # but we accept the input directly for simplicity.
        # Support both: {input: {prompt: ...}} and {prompt: ...}
        if "input" in body:
            job = body
        else:
            job = {"input": body}

        LOGGER.info("Processing request: %s", json.dumps(job.get("input", {}), default=str)[:200])
        start = time.time()
        result = handler(job)
        elapsed = time.time() - start
        LOGGER.info("Request completed in %.1fs, ok=%s", elapsed, result.get("ok"))

        status_code = 200 if result.get("ok") else 500
        self.send_json(status_code, result)

    def send_json(self, code, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        LOGGER.info(format, *args)


def preload_assets():
    """Download model weights on startup so first request is fast."""
    global _ready, _loading
    try:
        LOGGER.info("Pre-loading model assets...")
        ensure_assets()
        _ready = True
        _loading = False
        LOGGER.info("Model assets loaded — ready for requests")
    except Exception as e:
        _loading = False
        LOGGER.error("Failed to pre-load assets: %s", e)


if __name__ == "__main__":
    # Start asset download in background so HTTP server is immediately reachable
    threading.Thread(target=preload_assets, daemon=True).start()

    server = HTTPServer(("0.0.0.0", HTTP_PORT), LTXHandler)
    LOGGER.info("LTX HTTP server listening on port %d", HTTP_PORT)
    server.serve_forever()
