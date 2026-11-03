"""Сборка индекса и копирование моделей, нужных для поиска."""

import argparse
import json
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

from rag.common import digest, load_config, read_jsonl, relative
from rag.formulas import build_formula_index
from rag.retrieval import SearchIndex
from rag.indexing import INDEX_VERSION, build_passages, build_sections, fit_search_windows, named_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    settings = load_config(args.config)
    blocks = read_jsonl(args.corpus)
    seen = set()
    passages = []
    if not blocks:
        raise ValueError("Corpus is empty")
    for block in blocks:
        if (
            not isinstance(block.get("block_id"), str)
            or not block["block_id"]
            or block["block_id"] in seen
        ):
            raise ValueError("Invalid/duplicate corpus block ID")
        if (
            type(block.get("page")) is not int
            or block["page"] < 1
            or not isinstance(block.get("section"), str)
            or not isinstance(block.get("text"), str)
            or not block["text"].strip()
        ):
            raise ValueError("Invalid corpus row")
        seen.add(block["block_id"])
    passages = build_passages(blocks, settings["retrieval"]["chunk_chars"])
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = relative(settings, "tokenizer_dir")
    for name in ("tokenizer.json", "chat_template.jinja"):
        if not (tokenizer_dir / name).is_file():
            raise FileNotFoundError(
                f"Missing {tokenizer_dir / name}; run python -m scripts.prepare assets before the offline build"
            )
        shutil.copyfile(tokenizer_dir / name, output_dir / name)
    import numpy as np
    from rag.dense import Encoder

    embedding_config = settings["embedding"]
    embedding_dir = Path(settings["_root"]) / embedding_config["path"]
    encoder = Encoder(embedding_dir)
    passages = fit_search_windows(passages, encoder.tokenizer)
    sections = build_sections(blocks)
    max_embedding_tokens = max(len(ids) for ids in encoder.tokenizer(
        ["passage: " + passage["search_text"] for passage in passages],
        truncation=False, verbose=False,
    )["input_ids"])
    if max_embedding_tokens > embedding_config["max_length"]:
        raise ValueError("Search window exceeds the embedding model token limit")
    vectors = encoder.encode(
        [passage["search_text"] for passage in passages],
        "passage",
        embedding_config["batch_size"],
    )
    np.save(output_dir / "vectors.npy", vectors)
    np.save(output_dir / "section_vectors.npy", encoder.encode([section["title"] for section in sections], "passage"))
    # Готовый индекс должен работать без исходной папки скачанных моделей.
    (output_dir / "embedding").mkdir(exist_ok=True)
    for file in embedding_dir.iterdir():
        if file.is_file() and file.suffix in (".json", ".safetensors", ".model"):
            shutil.copyfile(file, output_dir / "embedding" / file.name)
    reranker_config = settings["reranker"]
    reranker_dir = Path(settings["_root"]) / reranker_config["path"]
    from rag.rerank import Reranker

    reranker = Reranker(reranker_dir)
    (output_dir / "reranker").mkdir(exist_ok=True)
    for file in reranker_dir.iterdir():
        if file.is_file() and file.suffix in (".json", ".safetensors", ".model"):
            shutil.copyfile(file, output_dir / "reranker" / file.name)
    index = SearchIndex(passages, settings["retrieval"])
    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "version": INDEX_VERSION,
                "blocks": blocks,
                "sections": sections,
                "named_results": named_results(blocks),
                "formula_index": build_formula_index(blocks),
                "corpus_sha256": digest(args.corpus),
                "retrieval": settings["retrieval"],
                "embedding": settings["embedding"],
                "reranker": settings["reranker"],
                "passages": passages,
            },
            ensure_ascii=False,
        )
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = {
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "size_bytes": sum(
            file.stat().st_size for file in output_dir.rglob("*") if file.is_file()
        ),
        "reranker_parameters": reranker.parameters,
        "blocks": len(blocks),
        "passages": len(index.passages),
        "formula_views": sum(p["view_kind"] == "formula" for p in passages),
        "sections": len(sections),
        "formula_records": len(build_formula_index(blocks)),
        "max_embedding_tokens": max_embedding_tokens,
        "index_version": INDEX_VERSION,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "corpus_sha256": digest(args.corpus),
    }
    (output_dir / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
