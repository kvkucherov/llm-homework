"""Загрузка моделей и учебника, извлечение корпуса и подготовка вопросов.

Сеть нужна только на этапе assets. Остальные этапы работают с локальными файлами."""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from rag.common import (
    check_questions,
    digest,
    load_config,
    read_jsonl,
    relative,
    write_jsonl,
)

PDF_SHA256 = "629f4b668ff43abf2a6e621aefd0f50e165ce8e88c52c05769149e1267d033ec"


def prepare_assets(settings, pdf_path, include_model=False):
    from huggingface_hub import snapshot_download

    root = Path(settings["_root"])
    runtime = json.loads(relative(settings, "runtime").read_text())
    snapshot_download(
        runtime["model"],
        revision=runtime["revision"],
        local_dir=relative(settings, "tokenizer_dir"),
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
    )
    for model in (settings["embedding"], settings["reranker"]):
        snapshot_download(
            model["model"],
            revision=model["revision"],
            local_dir=root / model["path"],
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "sentencepiece.bpe.model",
            ],
        )
    if include_model:
        snapshot_download(
            runtime["local_model"],
            revision=runtime["local_revision"],
            local_dir=root / runtime["model_dir"],
            allow_patterns=["*.json", "*.jinja", "*.safetensors"],
        )
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists():
        source = (root / "materials/homework_01_rag.md").read_text()
        url = re.search(r"\]\((https://new\.mipt\.ru/[^)]+)\)", source).group(1)
        urllib.request.urlretrieve(url, pdf_path)
    if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("The textbook PDF differs from the verified 2025 source")
    print("Assets ready. Index building and prediction need no downloads.")


def prepare_corpus(pdf_path, corpus_path):
    import pymupdf
    from rag.pdf_extract import extract_page, page_blocks

    blocks, audit, section = [], [], "Оглавление"
    with pymupdf.open(pdf_path) as doc:
        pages = len(doc)
        for number, page in enumerate(doc, 1):
            rows, stats = extract_page(page)
            page_result = page_blocks(rows, number, section)
            blocks.extend(page_result)
            if page_result:
                section = page_result[-1]["section"]
            audit.append({"page": number, **stats})
    write_jsonl(corpus_path, blocks)
    base = Path(corpus_path).parent
    write_jsonl(base / "extraction_audit.jsonl", audit)
    manifest = {
        "pdf_sha256": digest(pdf_path),
        "corpus_sha256": digest(corpus_path),
        "pages": pages,
        "blocks": len(blocks),
        "method": "font-and-geometry-v1; no OCR/LLM",
    }
    (base / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "pages": pages,
                "blocks": len(blocks),
                "fractions": sum(a["fractions"] for a in audit),
                "unsupported_glyphs": sum(
                    sum(a["unsupported"].values()) for a in audit
                ),
            }
        )
    )


# Страница PDF и начало фрагмента для эталонных ссылок открытого набора.
SOURCES = {
    1: [(11, "Теорема 2.1.")],
    2: [(12, "и мы приходим")],
    3: [(11, "Теорема 2.1."), (13, "Определение 2.3.")],
    4: [(26, "Теорема 4.1."), (28, "r"), (31, "Замечание 4.1.")],
    5: [(30, "Теорема 4.4.")],
    6: [(31, "Теорема 5.1"), (33, "Следствие 5.1.")],
    7: [(20, "Повторяя этот процесс"), (34, "Теорема 5.3.")],
    8: [(35, "Теорема 5.4")],
    9: [(11, "Теорема 2.1."), (13, "Система уравнений"), (34, "Теорема 5.3.")],
    10: [(36, "Теорема 6.1.")],
    11: [(37, "Теорема 6.2.")],
    12: [
        (21, "Тригонометрические функции."),
        (37, "Определение 6.1."),
        (38, "Теорема 6.3"),
    ],
    13: [(37, "Следствие 6.1")],
    14: [(26, "Аддитивность."), (28, "r")],
    15: [(20, "Повторяя этот процесс"), (36, "Теорема 6.1."), (37, "Определение 6.1.")],
}


def resolve(blocks, page, anchor):
    """Найти единственный блок по странице и текстовой метке из разметки."""
    found = [
        block
        for block in blocks
        if block["page"] == page
        and (
            "радиус окружности" in block["text"]
            if anchor == "r"
            else block["text"].startswith(anchor)
        )
    ]
    if len(found) != 1:
        raise ValueError(f"Ambiguous/missing gold anchor: {page}: {anchor}")
    return found[0]["block_id"]


def prepare_dev_questions(corpus_path, qa_path, output_dir):
    blocks = read_jsonl(corpus_path)
    questions = []
    gold = []
    parts = re.split(r"(?m)^## (\d+)\. [^\n]+\n", Path(qa_path).read_text())
    for i in range(1, len(parts), 2):
        number = int(parts[i])
        body = parts[i + 1]
        question_id = f"q_{number:03d}"
        category = re.search(r"\*\*Тип:\*\* (.+?)\.", body).group(1)
        question = (
            re.search(r"\*\*Вопрос\.\*\* (.*?)(?=\n- \*\*A\.)", body, re.S)
            .group(1)
            .strip()
        )
        options = [
            {"id": match[0], "text": match[1].strip()}
            for match in re.findall(r"(?m)^- \*\*([A-Z])\.\*\* (.+)$", body)
        ]
        answer = re.search(r"\*\*Ответ:\*\* ([^\n]+)", body).group(1)
        abstain = answer.startswith("Отказ")
        rationale = (
            body.split("**Ответ:**", 1)[1]
            .split("\n", 1)[1]
            .split("**Источник:**")[0]
            .strip()
        )
        sources = [resolve(blocks, *ref) for ref in SOURCES[number]]
        questions.append({"id": question_id, "question": question, "options": options})
        gold.append(
            {
                "id": question_id,
                "category": category,
                "answer_ids": [] if abstain else re.findall(r"\b[A-Z]\b", answer),
                "abstain": abstain,
                "citation_sets": [] if abstain else [sources],
                "reference_block_ids": sources,
                "rationale": rationale,
                **(
                    {"missing_information": rationale.split(". ")[0]} if abstain else {}
                ),
            }
        )
    if len(questions) != 15:
        raise ValueError("Expected 15 QA")
    check_questions(questions)
    destination = Path(output_dir)
    write_jsonl(destination / "dev_questions.jsonl", questions)
    write_jsonl(destination / "dev_gold.jsonl", gold)
    print("Prepared 15 questions and gold annotations.")


def prepare_own_questions(corpus_path, specs_path, output_dir):
    blocks = read_jsonl(corpus_path)
    questions = []
    gold = []
    for number, (category, question, options, answers, refs, reason) in enumerate(
        json.loads(Path(specs_path).read_text()), 1
    ):
        question_id = f"own_{number:03d}"
        sources = [resolve(blocks, *ref) for ref in refs]
        questions.append(
            {
                "id": question_id,
                "question": question,
                "options": [
                    {"id": chr(65 + i), "text": v} for i, v in enumerate(options)
                ],
            }
        )
        gold.append(
            {
                "id": question_id,
                "category": category,
                "answer_ids": answers,
                "abstain": not answers,
                "citation_sets": [sources] if answers else [],
                "reference_block_ids": sources,
                "rationale": reason,
                **({"missing_information": reason} if not answers else {}),
            }
        )
    check_questions(questions)
    write_jsonl(Path(output_dir) / "own_questions.jsonl", questions)
    write_jsonl(Path(output_dir) / "own_gold.jsonl", gold)
    print("Prepared 8 original questions and gold annotations.")


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка файлов задания без вызовов LLM."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", "assets", "corpus", "qa"],
        help="all: все этапы; assets: загрузка; corpus: PDF; qa: оба набора вопросов",
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--model",
        action="store_true",
        help="Также скачать Qwen (~3 ГБ), только для all/assets",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="PDF; по умолчанию local/data/textbook.pdf рядом с config",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Корпус; по умолчанию local/data/corpus.jsonl рядом с config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Общий каталог QA; по умолчанию dev в datasets/dev, авторские в корне",
    )
    args = parser.parse_args()
    if args.model and args.stage not in ("all", "assets"):
        parser.error("--model применяется только к all или assets")
    settings = load_config(args.config)
    root = Path(settings["_root"])
    pdf = args.pdf or root / "local/data/textbook.pdf"
    corpus = args.corpus or root / "local/data/corpus.jsonl"
    output = args.output_dir or root
    if args.stage in ("all", "assets"):
        prepare_assets(settings, pdf, args.model)
    if args.stage in ("all", "corpus"):
        prepare_corpus(pdf, corpus)
    if args.stage in ("all", "qa"):
        prepare_dev_questions(
            corpus,
            root / "materials/tfkp_15_qa.md",
            args.output_dir or root / "datasets/dev",
        )
        prepare_own_questions(
            corpus, root / "materials/own_question_specs.json", output
        )


if __name__ == "__main__":
    main()
