#!/usr/bin/env python3
"""Regression pins for the openai_compat request path in generate_answers.py.

Two things are pinned here, both cross-repo safety properties:

1. CROSS-REPO NO-OP. GENERATION_MAX_TOKENS / GENERATION_REASONING_EFFORT /
   GENERATION_PROVIDER_JSON must be absent from the request payload when their env vars are
   unset, so every banked run in every repo on this subtree stays byte-identical. The ollama
   path must not see them at all.

2. TRANSIENT-vs-FATAL CLASSIFICATION. OpenRouter reports upstream rate limits as **HTTP 200
   with {"error": {"code": 429, ...}} in the BODY**, which the HTTPError branch cannot see.
   Before TransientChatAPIError existed, such a response raised a plain RuntimeError,
   _is_retryable_request_error returned False, and the call failed having used ZERO of its
   retries — measured 2026-07-30 on a hosted GPT-5.6 run: 59 of 119 topics lost in one pass.
   The opposite failure matters just as much: a genuine 400 (bad model id) must STAY fatal, or
   a typo turns into a retry loop.

Run with plain python (no pytest needed):
    python generation/test_openai_compat_knobs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests  # noqa: E402
import generate_answers as G  # noqa: E402

_KNOBS = ("GENERATION_MAX_TOKENS", "GENERATION_REASONING_EFFORT", "GENERATION_PROVIDER_JSON")


class _FakeResp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def main() -> None:
    captured: dict = {}
    body: dict = {"choices": [{"message": {"content": "hi"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.clear()
        captured["payload"] = json
        return _FakeResp(body)

    G.requests.post = fake_post

    # 1. env absent -> the pre-change payload shape, exactly
    for k in _KNOBS:
        os.environ.pop(k, None)
    G.call_llm_openai_compat("k", "https://x/v1", "sys", "usr", model="m")
    assert set(captured["payload"]) == {"model", "messages", "temperature"}, captured["payload"]

    # 2. each knob forwarded, with the right type/shape, only when set
    os.environ["GENERATION_MAX_TOKENS"] = "16000"
    os.environ["GENERATION_REASONING_EFFORT"] = "medium"
    os.environ["GENERATION_PROVIDER_JSON"] = '{"only":["openai/flex"],"allow_fallbacks":false}'
    G.call_llm_openai_compat("k", "https://x/v1", "sys", "usr", model="m")
    p = captured["payload"]
    assert p["max_tokens"] == 16000, p
    assert p["reasoning"] == {"effort": "medium"}, p
    assert p["provider"] == {"only": ["openai/flex"], "allow_fallbacks": False}, p

    # 3. in-body 429 -> retryable
    body = {"error": {"code": 429, "message": "temporarily rate-limited upstream"}}
    try:
        G.call_llm_openai_compat("k", "https://x/v1", "s", "u", model="m")
        raise AssertionError("in-body 429 did not raise")
    except G.TransientChatAPIError as exc:
        assert G._is_retryable_request_error(exc) is True

    # 3b. marker-only match (no numeric code) still classifies as transient
    body = {"error": {"message": "Provider is overloaded, please try again"}}
    try:
        G.call_llm_openai_compat("k", "https://x/v1", "s", "u", model="m")
        raise AssertionError("overloaded body did not raise")
    except G.TransientChatAPIError:
        pass

    # 4. a genuine hard error STAYS fatal — the guard against retry-looping on a typo
    body = {"error": {"code": 400, "message": "invalid model id"}}
    try:
        G.call_llm_openai_compat("k", "https://x/v1", "s", "u", model="m")
        raise AssertionError("hard 400 did not raise")
    except G.TransientChatAPIError:
        raise AssertionError("hard 400 must not be classified transient")
    except RuntimeError as exc:
        assert G._is_retryable_request_error(exc) is False

    # 5. the pre-existing real-HTTP-429 path still works
    resp = requests.Response()
    resp.status_code = 429
    assert G._is_retryable_request_error(requests.exceptions.HTTPError(response=resp)) is True

    # 6. the ollama path never sees the openai_compat knobs (they are still set in env here)
    G.call_llm_ollama("k", "sys", "usr", model="gemma4:31b")
    assert not ({"reasoning", "provider", "max_tokens"} & set(captured["payload"])), captured

    for k in _KNOBS:
        os.environ.pop(k, None)
    print("test_openai_compat_knobs: OK")


if __name__ == "__main__":
    main()
