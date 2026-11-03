"""Запуск итогового варианта с локальным сервером Qwen."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from rag.common import check_questions, read_jsonl

ROOT = Path(__file__).resolve().parent


def ready(base, runtime):
    try:
        with urllib.request.urlopen(base + "/models", timeout=2) as response:
            state = json.load(response)
    except urllib.error.URLError:
        return False
    expected = {key: runtime[key] for key in ("model", "revision", "local_revision")}
    if state.get("status") != "ready" or any(
        state.get(k) != v for k, v in expected.items()
    ):
        raise RuntimeError("На локальном порту запущен другой сервер модели.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Запуск только лучшего варианта RAG.")
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "datasets/dev/dev_questions.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "local/outputs")
    parser.add_argument("--index", type=Path, default=ROOT / "local/index")
    args = parser.parse_args()
    questions, output = args.questions.resolve(), args.output_dir.resolve()
    questions_rows = read_jsonl(questions)
    check_questions(questions_rows)
    if not questions_rows:
        raise ValueError("Файл вопросов пуст.")
    settings = json.loads((ROOT / "config.json").read_text())
    runtime = json.loads((ROOT / settings["runtime"]).read_text())
    index = args.index.resolve()
    if not (index / "index.json").is_file():
        raise FileNotFoundError(
            "Индекс не подготовлен. Выполните раздел подготовки в README.md."
        )
    base = f"http://127.0.0.1:{runtime['port']}/v1"
    output.mkdir(parents=True, exist_ok=True)
    process, log = None, None
    try:
        if not ready(base, runtime):
            if not (ROOT / runtime["model_dir"] / "model.safetensors").is_file():
                raise FileNotFoundError(
                    "Сначала выполните python -m scripts.prepare assets --model."
                )
            log = (output / "server.log").open("w")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.serve_local",
                    "--config",
                    str(ROOT / "config.json"),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            print("Загружаю локальную Qwen…", flush=True)
            deadline = time.monotonic() + 60
            while not ready(base, runtime):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Сервер завершился: см. {output / 'server.log'}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Сервер не готов: см. {output / 'server.log'}")
                time.sleep(0.5)
        env = os.environ.copy()
        env.update(
            LLM_BASE_URL=base,
            LLM_MODEL=runtime["model"],
            SUBMISSION_ID="tfkp-final-" + uuid.uuid4().hex[:12],
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "predict.py"),
                "--index",
                str(index),
                "--questions",
                str(questions),
                "--output",
                str(output / "answers.jsonl"),
                "--trace",
                str(output / "trace.jsonl"),
                "--config",
                str(ROOT / "config.json"),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        failures = [
            trace_row["id"]
            for trace_row in read_jsonl(output / "trace.jsonl")
            if trace_row.get("error")
        ]
        if failures:
            raise RuntimeError(
                f"Технические ошибки в {failures}; подробности в trace.jsonl."
            )
        print(
            f"Ответы: {output / 'answers.jsonl'}\nТрассировка: {output / 'trace.jsonl'}"
        )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
