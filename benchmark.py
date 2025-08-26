#!/usr/bin/env python3
"""Benchmark tokens/sec against an OpenAI-compatible server (llama.cpp llama-server).

Usage: provide host, port, api-key, model, and prompt. Supports chat and completions endpoints and streaming.
"""
import argparse
import json
import re
import sys
import time
from typing import Optional

try:
    import requests
except Exception:
    print("requests is required. Install with: pip install requests", file=sys.stderr)
    raise

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False


def token_count(text: str, model: Optional[str] = None) -> int:
    """Return token count for text. Prefer tiktoken if available, else use simple heuristic."""
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            # encoding_for_model may fail for unknown models; fall back to cl100k_base
            try:
                enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic fallback: split into word-like tokens and punctuation
    tokens = re.findall(r"\w+|[^\s\w]", text)
    return len(tokens)


def build_url(host: str, port: int, endpoint: str) -> str:
    return f"http://{host}:{port}/v1/{endpoint}"


def run_non_streaming(url: str, headers: dict, payload: dict, model: str) -> None:
    start = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    elapsed = time.time() - start
    r.raise_for_status()
    data = r.json()

    # Extract response text depending on endpoint
    text = ""
    if isinstance(data, dict):
        # chat completions
        if data.get("choices"):
            choice = data["choices"][0]
            # chat
            if "message" in choice and "content" in choice["message"]:
                text = choice["message"]["content"]
            # completions
            elif "text" in choice:
                text = choice["text"]
        # some servers provide usage
        usage_tokens = None
        if "usage" in data and isinstance(data["usage"], dict):
            usage_tokens = data["usage"].get("total_tokens")

    num_tokens = usage_tokens if usage_tokens is not None else token_count(text, model=model)
    tps = num_tokens / elapsed if elapsed > 0 else float("inf")

    print(json.dumps({
        "mode": "non-stream",
        "tokens": num_tokens,
        "elapsed_seconds": round(elapsed, 6),
        "tokens_per_second": round(tps, 6),
    }, indent=2))


def run_streaming(url: str, headers: dict, payload: dict, model: str) -> None:
    # Use requests stream to process server-sent events or newline-delimited JSON
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()

        first_token_time = None
        last_token_time = None
        chunks = []

        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            # OpenAI-style streaming sends 'data: <json>' lines and 'data: [DONE]'
            if line.startswith("data:"):
                payload_text = line[len("data:"):].strip()
            else:
                payload_text = line

            if payload_text == "[DONE]":
                last_token_time = time.time()
                break

            try:
                obj = json.loads(payload_text)
            except Exception:
                # Not JSON — accumulate raw
                if payload_text:
                    chunks.append(payload_text)
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                    last_token_time = now
                continue

            # Extract text delta depending on chat/completions
            text_piece = ""
            try:
                choices = obj.get("choices") or []
                if choices:
                    ch = choices[0]
                    # chat streaming: delta.content
                    if "delta" in ch and isinstance(ch["delta"], dict):
                        text_piece = ch["delta"].get("content", "")
                    # completions streaming: text
                    elif "text" in ch:
                        text_piece = ch.get("text", "")
            except Exception:
                pass

            if text_piece:
                chunks.append(text_piece)
                now = time.time()
                if first_token_time is None:
                    first_token_time = now
                last_token_time = now

        full_text = "".join(chunks)
        elapsed = (last_token_time - first_token_time) if (first_token_time and last_token_time) else 0.0
        num_tokens = token_count(full_text, model=model)
        tps = num_tokens / elapsed if elapsed > 0 else float("inf")

        print(json.dumps({
            "mode": "stream",
            "tokens": num_tokens,
            "elapsed_seconds": round(elapsed, 6),
            "tokens_per_second": round(tps, 6),
        }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Benchmark tokens/sec against an OpenAI-compatible server")
    parser.add_argument("--host", default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=int, default=8080, help="server port")
    parser.add_argument("--api-key", required=True, help="API key for Authorization header")
    parser.add_argument("--model", required=True, help="Model id")
    parser.add_argument("--prompt", required=True, help="Prompt or message to send")
    parser.add_argument("--endpoint", choices=["chat/completions", "completions"], default="chat/completions",
                        help="Which endpoint path to call (default: chat/completions)")
    parser.add_argument("--stream", action="store_true", help="Use streaming responses (if the server supports it)")
    parser.add_argument("--max-tokens", type=int, default=512, help="max_tokens for the response")
    args = parser.parse_args()

    host = args.host
    port = args.port
    api_key = args.api_key
    model = args.model
    prompt = args.prompt
    endpoint_path = args.endpoint

    url = build_url(host, port, endpoint_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build payload depending on endpoint
    if endpoint_path == "chat/completions":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": args.stream,
            "max_tokens": args.max_tokens,
        }
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": args.stream,
            "max_tokens": args.max_tokens,
        }

    # Run appropriate mode
    if args.stream:
        run_streaming(url, headers, payload, model)
    else:
        run_non_streaming(url, headers, payload, model)


if __name__ == "__main__":
    main()
