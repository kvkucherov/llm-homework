"""Структурный поиск формул; правила не зависят от вопросов или их эталонов."""

import re

from rag.indexing import RESULT, normalize_math

ATOM = r"(?:\\[A-Za-z]+|[A-Za-z])(?:_(?:\{[^{}]+\}|[A-Za-z0-9]))?"
BASE = rf"(?:{ATOM}|\({ATOM}(?:[-+](?:{ATOM}|\d+))?\))"
POWER = rf"{BASE}\^(?:\{{(?P<braced>[^{{}}]+)\}}|(?P<bare>[0-9]+|[A-Za-z]))"
FUNCTION = r"(\\[A-Za-z]+|[A-Za-z])"


def _group(text, start):
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start + 1:end], end + 1
    return None


def _fractions(text):
    for match in re.finditer(r"\\frac\b", text):
        numerator = _group(text, match.end())
        if numerator is not None:
            denominator = _group(text, numerator[1])
            if denominator is not None:
                yield numerator[0], denominator[0]


def integral_signatures(text):
    """Порядок степенного знаменателя, факториал и связь с производной."""
    normalized = normalize_math(text)
    integral_in_text = bool(re.search(r"\b(?:интеграл[а-я]*|integrals?)\b", normalized.lower()))
    expressions = re.findall(r"\$([^$]+)\$", normalized) or [normalized]
    signatures = []
    for expression in expressions:
        expression = re.sub(r"\s+", "", expression)
        if not integral_in_text and not re.search(r"\\(?:int|oint)(?![A-Za-z])", expression):
            continue
        fractions = list(_fractions(expression))
        powers = [re.fullmatch(POWER, denominator) for _, denominator in fractions]
        powers += list(re.finditer(r"/" + POWER, expression))
        lhs = expression.split("=", 1)[0]
        derivatives = set(re.findall(
            FUNCTION + r"(?:\^\{?\([A-Za-z0-9]+\)\}?|'+)\(", lhs
        ))
        numerator_functions = {
            function for numerator, _ in fractions
            for function in re.findall(FUNCTION + r"\(", numerator)
        }
        for power in powers:
            if power is None:
                continue
            degree = power["braced"] or power["bare"]
            if degree.isdigit():
                degree = int(degree)
                if degree < 2:
                    continue
            elif re.fullmatch(r"[A-Za-z](?:[+-]\d+)?", degree):
                degree = "symbolic"
            else:
                continue
            signature = {
                "degree": degree,
                "factorial": bool(re.search(r"(?:[A-Za-z0-9]|\([^()]+\))!", expression)),
                "same_function_derivative": bool(derivatives & numerator_functions),
            }
            if signature not in signatures:
                signatures.append(signature)
    return signatures


def build_formula_index(blocks):
    records = []
    for block in blocks:
        for signature in integral_signatures(block["text"]):
            records.append({
                "block_id": block["block_id"],
                "statement": bool(RESULT.match(block["text"])),
                **signature,
            })
    return records


def formula_sources(question, records):
    degrees = {item["degree"] for item in integral_signatures(question)}
    if not degrees:
        return []
    matching = [
        row for row in records
        if row["factorial"] and row["same_function_derivative"]
        and (row["degree"] == "symbolic" or row["degree"] in degrees)
    ]
    # Предпочитаем сформулированный общий результат, а не шаг доказательства.
    matching.sort(key=lambda row: (not row["statement"], row["degree"] != "symbolic"))
    return [matching[0]["block_id"]] if matching else []
