# llm-oai-tpsbench

A small command-line benchmark tool to measure tokens-per-second (TPS) from an OpenAI-compatible server (for example, llama.cpp/llama-server or other OpenAI-compatible endpoints).

The repository contains a single script, `benchmark.py`, which calls either the `chat/completions` or `completions` endpoints, supports streaming, and reports tokens, elapsed time, and tokens/sec.

## Features

- Works with OpenAI-compatible HTTP APIs (local or remote).
- Supports chat-style (`chat/completions`) and classic completions endpoints.
- Supports streaming responses (server must support OpenAI-style streaming lines).
- Uses `tiktoken` for accurate token counts when available; falls back to a simple heuristic otherwise.

## Requirements

- Python 3.8+ recommended
- `requests` (required)
- `tiktoken` (optional — improves token counting accuracy)

Install dependencies:

```bash
pip install requests
# optional, for better token counting
pip install tiktoken
```

## Usage

Basic help:

```bash
python3 benchmark.py --help
```

Typical non-streaming run:

```bash
python3 benchmark.py \
  --api-key YOUR_API_KEY \
  --model your-model-id \
  --prompt "Write a short poem about benchmarking." \
  --host 127.0.0.1 \
  --port 8080
```

Streaming example (if the server supports streaming):

```bash
python3 benchmark.py \
  --api-key YOUR_API_KEY \
  --model your-model-id \
  --prompt "Stream me a sentence" \
  --host 127.0.0.1 \
  --port 8080 \
  --stream
```

If you want to test the classic completions endpoint instead of chat-style calls:

```bash
python3 benchmark.py --endpoint completions --api-key ... --model ... --prompt "Say hi"
```

## Output

The script prints a JSON object with these fields:

- `mode`: `stream` or `non-stream`
- `tokens`: number of tokens measured in the response (uses server `usage.total_tokens` if present, otherwise local count)
- `elapsed_seconds`: measured elapsed time (stream: time between first and last received token)
- `tokens_per_second`: computed tokens/sec

Example output:

```json
{
  "mode": "non-stream",
  "tokens": 120,
  "elapsed_seconds": 0.534,
  "tokens_per_second": 224.719
}
```

## How token counting works

- If `tiktoken` is installed the script will attempt to use the encoding for the specified model (or fall back to `cl100k_base`) for accurate counts.
- If `tiktoken` is not available, the script uses a simple heuristic that splits on words and punctuation. This is less accurate but still useful for rough benchmarking.

