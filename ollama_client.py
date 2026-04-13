"""Thin wrapper around the Ollama HTTP API.

Ollama must be running locally (default: http://localhost:11434).
Every generate() call returns timing and token stats extracted from
the Ollama response body — no extra instrumentation needed.
"""

import time
import requests

OLLAMA_BASE = "http://localhost:11434"
_TIMEOUT = 300  # seconds — large models can be slow


def list_models() -> list[str]:
    """Return names of models currently pulled in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def is_running() -> bool:
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def generate(prompt: str, model: str, system: str | None = None) -> dict:
    """Run a prompt against a local Ollama model using streaming.

    Streaming is used so large models (e.g. 26B) never cause a read
    timeout — tokens arrive continuously rather than all at once.

    Returns a dict with:
        response               str   — full model output text
        input_tokens           int   — tokens in the prompt
        output_tokens          int   — tokens generated
        total_duration_ms      float — wall time reported by Ollama (ms)
        generation_duration_ms float — pure generation time (ms)
        tokens_per_second      float — output tok/s during generation
        wall_time_ms           float — actual HTTP round-trip time (ms)
    """
    import json as _json

    payload: dict = {"model": model, "prompt": prompt, "stream": True}
    if system:
        payload["system"] = system

    t0 = time.perf_counter()
    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json=payload,
        stream=True,
        # connect timeout 30 s; read timeout per chunk 120 s
        timeout=(30, 120),
    )
    r.raise_for_status()

    chunks = []
    final: dict = {}

    for raw_line in r.iter_lines():
        if not raw_line:
            continue
        try:
            obj = _json.loads(raw_line)
        except _json.JSONDecodeError:
            continue

        if obj.get("response"):
            chunks.append(obj["response"])

        if obj.get("done"):
            final = obj  # last line contains all the stats

    wall_time_ms = (time.perf_counter() - t0) * 1000

    input_tokens = final.get("prompt_eval_count", 0)
    output_tokens = final.get("eval_count", 0)
    total_duration_ms = final.get("total_duration", 0) / 1_000_000
    gen_duration_ms = final.get("eval_duration", 0) / 1_000_000
    tps = (output_tokens / (gen_duration_ms / 1000)) if gen_duration_ms > 0 else 0.0

    return {
        "response": "".join(chunks),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_duration_ms": round(total_duration_ms, 2),
        "generation_duration_ms": round(gen_duration_ms, 2),
        "tokens_per_second": round(tps, 2),
        "wall_time_ms": round(wall_time_ms, 2),
    }
