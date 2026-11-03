"""Локальный API для Qwen через MLX.

Сервер возвращает расход токенов и обрабатывает запросы без общей истории."""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from rag.common import load_config, relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    settings = load_config(args.config)
    runtime = json.loads(relative(settings, "runtime").read_text())
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    model_path = Path(settings["_root"]) / runtime["model_dir"]
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(
            "Download weights first: python -m scripts.prepare assets --model"
        )
    model, tokenizer = load(
        str(model_path), tokenizer_config={"trust_remote_code": False}
    )
    # Токенизатор должен совпадать со счётчиком перед запросом к серверу.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(relative(settings, "tokenizer_dir")),
        local_files_only=True,
        trust_remote_code=False,
    )

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send_json(
                200,
                {
                    "status": "ready",
                    "model": runtime["model"],
                    "revision": runtime["revision"],
                    "local_revision": runtime["local_revision"],
                },
            )

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.send_json(404, {"error": "Unknown endpoint"})
            try:
                request = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                )
                if not self.headers.get("X-Question-ID") or not self.headers.get(
                    "X-Submission-ID"
                ):
                    raise ValueError("Required accounting headers missing")
                if request.get("model") != runtime["model"] or request.get("stream"):
                    raise ValueError("Model mismatch or unsupported streaming")
                for key, value in runtime["sampling"].items():
                    if request.get(key) != value:
                        raise ValueError(f"Fixed sampling parameter mismatch: {key}")
                if (
                    request.get("chat_template_kwargs", {}).get("enable_thinking")
                    != runtime["enable_thinking"]
                ):
                    raise ValueError("Thinking setting mismatch")
                prompt = tokenizer.apply_chat_template(
                    request["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=runtime["enable_thinking"],
                )
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                maximum = request["max_tokens"]
                if (
                    type(maximum) is not int
                    or not 1 <= maximum <= 4096
                    or len(input_ids) + maximum > 8192
                ):
                    raise ValueError("Request exceeds token budget")
            except Exception as exc:
                return self.send_json(400, {"error": str(exc)})
            try:
                mx.random.seed(runtime["sampling"]["seed"])
                chunks = []
                last = None
                start = time.monotonic()
                for part in stream_generate(
                    model,
                    tokenizer,
                    prompt=input_ids,
                    max_tokens=maximum,
                    sampler=make_sampler(
                        temp=runtime["sampling"]["temperature"],
                        top_p=runtime["sampling"]["top_p"],
                        top_k=runtime["sampling"].get("top_k", 0),
                    ),
                    logits_processors=make_logits_processors(
                        presence_penalty=runtime["sampling"].get("presence_penalty", 0),
                        presence_context_size=8192,
                    ),
                    prefill_step_size=512,
                ):
                    chunks.append(part.text)
                    last = part
                result = {
                    "id": "local-" + str(time.time_ns()),
                    "object": "chat.completion",
                    "model": runtime["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(chunks),
                            },
                            "finish_reason": last.finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": last.prompt_tokens,
                        "completion_tokens": last.generation_tokens,
                        "total_tokens": last.prompt_tokens + last.generation_tokens,
                    },
                }
                print(
                    json.dumps(
                        {
                            "question_id": self.headers["X-Question-ID"],
                            "submission_id": self.headers["X-Submission-ID"],
                            **result["usage"],
                            "elapsed_seconds": time.monotonic() - start,
                            "peak_memory_bytes": mx.get_peak_memory(),
                        }
                    ),
                    flush=True,
                )
                self.send_json(200, result)
                mx.clear_cache()
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})

    print(
        "READY http://127.0.0.1:" + str(args.port or runtime["port"]) + "/v1",
        flush=True,
    )
    HTTPServer(("127.0.0.1", args.port or runtime["port"]), Handler).serve_forever()


if __name__ == "__main__":
    main()
