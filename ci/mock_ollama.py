#!/usr/bin/env python3
"""Deterministic mock ollama /api/generate endpoint for pipeline CI/verification.

Routes on prompt content:
  extract prompts  (ask for {"claims": [...]})       -> two mock claims, unique per passage
  summarize prompts ("Combined statement:")          -> one mock summary, unique per facet
  anything else (answer generation)                  -> fixed answer JSON

No LLM, no network egress, fully deterministic -> byte-identical reruns.
Usage: python3 mock_ollama.py [port]
"""
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            prompt = json.loads(body).get("prompt", "")
        except Exception:
            prompt = ""
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        if '"claims"' in prompt:
            resp = json.dumps({"claims": [
                f"Mock claim {h} states one fact.",
                f"Mock claim {h} adds a second fact.",
            ]})
        elif "Combined statement:" in prompt:
            resp = f"Mock combined statement {h} preserving every fact."
        else:
            resp = json.dumps({"ideal_answer": "Mock answer.", "evidence_ids": []})
        out = json.dumps({"response": resp, "done": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
