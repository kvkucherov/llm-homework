"""Чтение данных, настройки и проверка формата вопросов и ответов."""

import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{i}: invalid JSON") from e
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    cfg["_root"] = str(path.parent)
    cfg["_path"] = str(path)
    return cfg


def relative(cfg, name):
    p = Path(cfg[name])
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_questions(rows):
    seen = set()
    for question in rows:
        if (
            not isinstance(question.get("id"), str)
            or not question["id"]
            or question["id"] in seen
        ):
            raise ValueError("Question IDs must be unique nonempty strings")
        seen.add(question["id"])
        if (
            not isinstance(question.get("question"), str)
            or not question["question"].strip()
        ):
            raise ValueError(f"Empty question: {question['id']}")
        options = question.get("options", [])
        if not 4 <= len(options) <= 6:
            raise ValueError(f"Expected 4–6 options: {question['id']}")
        ids = [option.get("id") for option in options]
        if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(
            ids
        ):
            raise ValueError(f"Invalid option IDs: {question['id']}")
        if any(
            not isinstance(option.get("text"), str) or not option["text"].strip()
            for option in options
        ):
            raise ValueError("Empty option")


def check_answer(answer, question, valid_blocks, no_context=False):
    expected = {"id", "answer_ids", "abstain", "rationale", "citations"}
    if not isinstance(answer, dict) or set(answer) != expected:
        raise ValueError("Expected exactly id,answer_ids,abstain,rationale,citations")
    if answer["id"] != question["id"] or type(answer["abstain"]) is not bool:
        raise ValueError("Wrong id/abstain type")
    for name in ("answer_ids", "citations"):
        values = answer[name]
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(f"{name} must be a list of strings")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name}")
    if not set(answer["answer_ids"]) <= {o["id"] for o in question["options"]}:
        raise ValueError("Unknown answer option")
    if (
        not set(answer["citations"]) <= set(valid_blocks)
        or len(answer["citations"]) > 4
    ):
        raise ValueError("Unknown or excessive citations")
    if (
        not isinstance(answer["rationale"], str)
        or not 1 <= len(answer["rationale"].strip()) <= 1200
    ):
        raise ValueError("Rationale must contain 1–1200 characters")
    if answer["abstain"]:
        if answer["answer_ids"] or answer["citations"]:
            raise ValueError("Abstention must have empty lists")
    elif not answer["answer_ids"] or (not answer["citations"] and not no_context):
        raise ValueError("An answer needs options and evidence")
    return answer
