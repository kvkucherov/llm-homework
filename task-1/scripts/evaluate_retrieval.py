"""Оценка собранного индекса по эталонным источникам без генерации ответов."""

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

from predict import pack_context
from rag.budget import TokenCounter
from rag.common import load_config, read_jsonl, relative
from rag.retrieval import SearchIndex


def summarize(gold, records):
    by_id = {row["id"]: row for row in records}
    answerable = [row for row in gold if not row["abstain"]]
    result = {}
    for stage in ("candidate_block_ids", "ranked_block_ids", "retrieved_block_ids"):
        recalls = []
        for row in answerable:
            actual = set(by_id[row["id"]][stage])
            recalls.append(max(
                len(actual & set(sources)) / len(set(sources))
                for sources in row["citation_sets"] if sources
            ))
        result[stage] = {
            "mean_recall": statistics.mean(recalls),
            "any_source_questions": sum(value > 0 for value in recalls),
            "complete_set_questions": sum(value == 1 for value in recalls),
            "answerable_questions": len(recalls),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "questions", "gold", "config"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    settings = load_config(args.config)
    runtime = json.loads(relative(settings, "runtime").read_text())
    index = SearchIndex.load(args.index)
    if index.config != settings["retrieval"]:
        raise ValueError("Index and retrieval config differ; rebuild the index")
    counter = TokenCounter(args.index)
    records = []
    start = time.monotonic()
    for question in read_jsonl(args.questions):
        candidates, detail = index.retrieve(question, "final")
        _, selected = pack_context(question, candidates, counter, settings, runtime)
        row = {"id": question["id"], **{
            key: detail[key] for key in ("candidate_block_ids", "ranked_block_ids")
        }, "retrieved_block_ids": list(dict.fromkeys(p["block_id"] for p in selected))}
        records.append(row)
        print(row["id"], row["retrieved_block_ids"], flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "metrics": summarize(read_jsonl(args.gold), records),
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "records": records,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
