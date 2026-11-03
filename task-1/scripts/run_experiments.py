"""Обязательные режимы и две абляции с одними параметрами Qwen."""

import argparse
import csv
import json
import os
import shutil
import uuid
from pathlib import Path

from predict import run
from rag.common import digest, load_config, read_jsonl, relative

from scripts.evaluate import score

NAMES = ["final", "no_context", "baseline", "no_reranker", "no_dense"]
FIELDS = "name,answer_accuracy,Q,mean_input_tokens,mean_output_tokens,mean_llm_calls,mean_cost,median_latency_ms".split(
    ","
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="local/index")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--questions", default="datasets/dev/dev_questions.jsonl")
    parser.add_argument("--gold", default="datasets/dev/dev_gold.jsonl")
    parser.add_argument("--corpus", default="local/data/corpus.jsonl")
    parser.add_argument("--output-dir", default="local/runs")
    parser.add_argument("--reuse-complete", action="store_true")
    args = parser.parse_args()
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    settings = load_config(args.config)
    signature = {
        "config": digest(args.config),
        "questions": digest(args.questions),
        "corpus": digest(args.corpus),
        "runtime": digest(relative(settings, "runtime")),
        "index": digest(Path(args.index) / "index.json"),
        "vectors": digest(Path(args.index) / "vectors.npy"),
        "section_vectors": digest(Path(args.index) / "section_vectors.npy"),
    }
    for file in (
        "predict.py",
        "build_index.py",
        "rag/indexing.py",
        "rag/formulas.py",
        "rag/retrieval.py",
        "rag/budget.py",
        "rag/dense.py",
        "rag/rerank.py",
        "rag/common.py",
        "scripts/serve_local.py",
        "requirements.txt",
    ):
        signature[file] = digest(Path(__file__).resolve().parents[1] / file)
    results = {}
    questions = read_jsonl(args.questions)
    submission_base = (
        os.environ.get("SUBMISSION_ID", "tfkp-local") + "-" + uuid.uuid4().hex[:10]
    )
    for name in NAMES:
        answers_path = folder / (name + "_answers.jsonl")
        trace_path = folder / (name + "_trace.jsonl")
        manifest_path = folder / (name + "_manifest.json")
        reusable = (
            args.reuse_complete
            and answers_path.exists()
            and trace_path.exists()
            and manifest_path.exists()
            and json.loads(manifest_path.read_text()) == signature
        )
        reusable = (
            reusable
            and len(read_jsonl(answers_path)) == len(questions)
            and len(read_jsonl(trace_path)) == len(questions)
        )
        if not reusable:
            os.environ["SUBMISSION_ID"] = submission_base + "-" + name
            print("EXPERIMENT", name, flush=True)
            run(args.index, args.questions, answers_path, trace_path, args.config, name)
            manifest_path.write_text(json.dumps(signature, indent=2))
        results[name] = score(
            questions,
            read_jsonl(args.gold),
            read_jsonl(answers_path),
            read_jsonl(trace_path),
            read_jsonl(args.corpus),
            name == "no_context",
        )
    baseline_cost = results["baseline"]["mean_cost"]
    for name, metrics in results.items():
        if (
            metrics["Q"] is not None
            and baseline_cost is not None
            and metrics["mean_cost"] is not None
        ):
            metrics["E"] = (
                min(1, baseline_cost / max(metrics["mean_cost"], 1))
                if metrics["efficiency_valid"]
                else 0
            )
            metrics["grade"] = 75 * metrics["Q"] + 25 * metrics["Q"] * metrics["E"]
    with open("experiments.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for name in NAMES:
            writer.writerow(
                {"name": name, **{key: results[name][key] for key in FIELDS[1:]}}
            )
    results["_provenance"] = {
        "signature": signature,
        "note": "All configurations use the current pipeline; completed runs may be reused only with matching signatures.",
        "submission_ids": {
            name: sorted(
                {
                    trace_row["submission_id"]
                    for trace_row in read_jsonl(folder / (name + "_trace.jsonl"))
                }
            )
            for name in NAMES
        },
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/experiment_details.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    for kind in ("answers", "trace"):
        shutil.copyfile(folder / f"final_{kind}.jsonl", f"dev_{kind}.jsonl")
    print(
        "Saved experiments.csv, results/experiment_details.json, dev_answers.jsonl, dev_trace.jsonl"
    )


if __name__ == "__main__":
    main()
