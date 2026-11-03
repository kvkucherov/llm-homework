"""Проверка ответов и расчёт метрик по формулам задания."""

import argparse
import json
import statistics
from collections import Counter, defaultdict

from rag.common import check_answer, check_questions, read_jsonl

CATEGORIES = ("условия", "сопоставление", "применение", "недостаточность данных")


def citation_f1(actual, allowed):
    """Взять лучший F1 среди допустимых наборов ссылок."""
    actual_ids = set(actual)
    return max(
        (
            2 * len(actual_ids & set(gold_ids)) / (len(actual_ids) + len(set(gold_ids)))
            if actual_ids or gold_ids
            else 1
            for gold_ids in allowed
        ),
        default=0,
    )


def score(questions, gold, answers, traces, corpus, no_context=False, c_ref=None):
    check_questions(questions)
    gold_by_id = {r["id"]: r for r in gold}
    answers_by_id = {r.get("id"): r for r in answers}
    traces_by_id = {r.get("id"): r for r in traces}
    expected = {question["id"] for question in questions}
    if set(gold_by_id) != expected:
        raise ValueError("Gold IDs must match question IDs")
    answer_counts = Counter(r.get("id") for r in answers)
    trace_counts = Counter(r.get("id") for r in traces)
    blocks = {b["block_id"] for b in corpus}
    by_category = defaultdict(list)
    details = []
    correct = []
    costs = []
    input_tokens = []
    output_tokens = []
    calls = []
    latencies = []
    extras = (set(answers_by_id) | set(traces_by_id)) - expected
    efficiency_valid = not extras
    for question in questions:
        question_id = question["id"]
        error = None
        answer = answers_by_id.get(question_id)
        trace = traces_by_id.get(question_id)
        reference = gold_by_id[question_id]
        try:
            if answer_counts[question_id] != 1 or trace_counts[question_id] != 1:
                raise ValueError("Missing or duplicate answer/trace")
            check_answer(answer, question, blocks, no_context)
            if trace.get("error"):
                raise ValueError(trace["error"])
        except Exception as exc:
            error = str(exc)
        valid = bool(trace) and trace.get("usage_verified") is True
        if trace:
            for field, maximum in (
                ("llm_calls", 4),
                ("input_tokens", 16384),
                ("output_tokens", 4096),
            ):
                value = trace.get(field)
                valid &= type(value) is int and 0 <= value <= maximum
            events = trace.get("requests", [])
            valid &= len(events) == trace.get("llm_calls")
            if valid:
                valid &= (
                    sum(event["input_tokens"] for event in events)
                    == trace["input_tokens"]
                    and sum(event["output_tokens"] for event in events)
                    == trace["output_tokens"]
                )
                valid &= all(
                    event["input_tokens"] + event["reserved_output_tokens"] <= 8192
                    and event["output_tokens"] <= event["reserved_output_tokens"]
                    for event in events
                )
        if not valid:
            efficiency_valid = False
            error = error or "Unverifiable/excessive token usage"
        answer_match = 0
        source_f1 = 0
        question_score = 0
        if not error:
            answer_match = int(
                answer["abstain"] == reference["abstain"]
                and set(answer["answer_ids"]) == set(reference["answer_ids"])
            )
            source_f1 = (
                citation_f1(answer["citations"], reference["citation_sets"])
                if not reference["abstain"]
                else 0
            )
            question_score = (
                answer_match
                if reference["abstain"]
                else answer_match * (0.8 + 0.2 * source_f1)
            )
        correct.append(answer_match)
        by_category[reference["category"]].append(question_score)
        details.append(
            {
                "id": question_id,
                "category": reference["category"],
                "A": answer_match,
                "F": source_f1,
                "score": question_score,
                "error": error,
            }
        )
        if valid:
            input_tokens.append(trace["input_tokens"])
            output_tokens.append(trace["output_tokens"])
            calls.append(trace["llm_calls"])
            costs.append(
                trace["input_tokens"]
                + 4 * trace["output_tokens"]
                + 256 * trace["llm_calls"]
            )
        if trace and isinstance(trace.get("elapsed_ms"), (int, float)):
            latencies.append(trace["elapsed_ms"])
    category_scores = {
        c: statistics.mean(by_category[c]) if by_category[c] else 0 for c in CATEGORIES
    }
    quality = statistics.mean(category_scores.values())

    def average(values):
        # Средний расход имеет смысл, только если он известен для всех вопросов.
        if len(values) != len(questions):
            return None
        return statistics.mean(values)

    mean_cost = average(costs)
    efficiency = (
        min(1, c_ref / max(mean_cost, 1))
        if efficiency_valid and c_ref is not None and mean_cost is not None
        else 0
    )
    return {
        "answer_accuracy": statistics.mean(correct),
        "Q": None if no_context else quality,
        "mean_input_tokens": average(input_tokens),
        "mean_output_tokens": average(output_tokens),
        "mean_llm_calls": average(calls),
        "mean_cost": mean_cost,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "efficiency_valid": efficiency_valid,
        "E": efficiency,
        "grade": None if no_context else 75 * quality + 25 * quality * efficiency,
        "category_scores": category_scores,
        "details": details,
        "unexpected_ids": sorted(extras),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("questions", "gold", "answers", "trace", "corpus"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--c-ref", type=float)
    parser.add_argument("--output")
    arguments = vars(parser.parse_args())
    output_path = arguments.pop("output")
    arguments["traces"] = read_jsonl(arguments.pop("trace"))
    for key in ("questions", "gold", "answers", "corpus"):
        arguments[key] = read_jsonl(arguments[key])
    text = json.dumps(score(**arguments), ensure_ascii=False, indent=2)
    if output_path:
        from pathlib import Path

        Path(output_path).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
