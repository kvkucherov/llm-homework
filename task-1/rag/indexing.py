"""Детерминированные поисковые представления корпуса, без LLM и QA-разметки."""

import re

INDEX_VERSION = 3
STATEMENT = re.compile(r"^(Теорема|Определение|Следствие|Лемма|Замечание|Пример)\s+\d+\.\d+")
RESULT = re.compile(r"^(Теорема|Следствие|Лемма)\s+\d+\.\d+")
SYMBOLS = {
    "∂": r"\partial ", "∫": r"\int ", "∑": r"\sum ",
    "π": r"\pi ", "∞": r"\infty ", "γ": r"\gamma ",
    "ζ": r"\zeta ", "−": "-", "′": "'", "″": "''",
    "≤": r"\le ", "≥": r"\ge ", "≡": r"\equiv ",
}


def normalize_math(text):
    for symbol, replacement in SYMBOLS.items():
        text = text.replace(symbol, replacement)
    text = re.sub(r"([a-zA-Z])\u0304", r"\\overline{\1}", text)
    text = re.sub(r"\\(?:tfrac|dfrac)\b", r"\\frac", text)
    text = re.sub(r"\\(?:tfrac|dfrac|frac)\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\bar\b", r"\\overline", text)
    text = re.sub(r"\\overline\s*([a-zA-Z])", r"\\overline{\1}", text)
    text = re.sub(r"\\(?:left|right)(?=[\s()\[\]|\\])", "", text)
    text = re.sub(r"\\(?:,|;|!|quad\b|qquad\b)", " ", text)
    return text


def notation_labels(text):
    """Названия синтаксических конструкций, одинаковые для документа и запроса."""
    labels = []
    compact = re.sub(r"\s+", "", normalize_math(text))
    if r"\partial" in compact:
        labels.append("частные производные")
    if r"\overline" in compact:
        labels.append("комплексное сопряжение")
    if re.search(r"[a-z](?:'+|\^\{?\([a-z0-9]+\)\}?)\(", compact):
        labels.append("производная")
    if re.search(r"[a-z0-9]!", compact):
        labels.append("факториал")
    if r"\int" in compact or r"\oint" in compact:
        labels.append("интеграл")
    if r"\sum" in compact and re.search(r"[a-z)]\^\{?[a-z]\}?", compact):
        labels.append("степенной ряд")
    if re.search(r"\be\s*\^|\\exp\b", normalize_math(text)):
        labels.append("экспонента")
    return labels


def search_text(text):
    normalized = normalize_math(text)
    labels = notation_labels(normalized)
    return normalized + ("\nОбозначения: " + "; ".join(labels) if labels else "")


def build_passages(blocks, chunk_chars):
    """Полный текст и локальные окна формул с неизменными исходными ссылками."""
    groups = {}
    current = []
    for block in blocks:
        text = block["text"].strip()
        if (
            not current
            or STATEMENT.match(text)
            or text.startswith(("Доказательство", "§"))
            or block["section"] != current[-1]["section"]
            or not 0 <= block["page"] - current[-1]["page"] <= 1
            or sum(len(b["text"]) for b in current) + len(text) > 3500
        ):
            current = [block] if STATEMENT.match(text) else []
        elif current:
            current.append(block)
        if current:
            for member in current:
                groups[member["block_id"]] = current

    passages = []
    for block in blocks:
        group = groups.get(block["block_id"], [block])
        title = group[0]["text"].splitlines()[0] if STATEMENT.match(group[0]["text"]) else ""
        lines = block["text"].splitlines()
        windows = []
        start, length = 0, 0
        for i, line in enumerate(lines):
            if length + len(line) + 1 > chunk_chars and i > start:
                windows.append((start, i, "text"))
                start, length = i, 0
            length += len(line) + 1
        windows.append((start, len(lines), "text"))
        formulas = [i for i, line in enumerate(lines) if line.lstrip().startswith("$")]
        for i in formulas:
            left, right = max(0, i - 2), min(len(lines), i + 2)
            if sum(len(line) for line in lines[left:right]) <= chunk_chars:
                windows.append((left, right, "formula"))
        seen = set()
        for left, right, kind in windows:
            excerpt = "\n".join(lines[left:right])
            if excerpt in seen:
                continue
            seen.add(excerpt)
            heading = title if title and not excerpt.startswith(title) else ""
            view = "\n".join(part for part in (block["section"], heading, excerpt) if part)
            passages.append({
                **block,
                "chunk": len(seen) - 1,
                "view_kind": kind,
                "search_text": search_text(view),
                "context_block_ids": [member["block_id"] for member in group],
            })
    return passages


def fit_search_windows(passages, tokenizer):
    """Не терять конец формулы из-за ограничения E5 в 512 токенов."""
    result = []
    for passage in passages:
        text = passage["search_text"]
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
        if len(offsets) <= 480:
            result.append(passage)
            continue
        for start in range(0, len(offsets), 448):
            end = min(start + 480, len(offsets))
            result.append({**passage, "search_text": text[offsets[start][0]:offsets[end - 1][1]]})
            if end == len(offsets):
                break
    return result


def build_sections(blocks):
    """Оглавление и формулировки результатов; доказательства не дублируются."""
    sections = {}
    for block in blocks:
        section = sections.setdefault(block["section"], {"title": block["section"], "result_block_ids": []})
        if RESULT.match(block["text"]):
            section["result_block_ids"].append(block["block_id"])
    return list(sections.values())


def named_results(blocks):
    """Явные названия теорем из учебника для точного поиска по названию."""
    names = []
    for block in blocks:
        match = re.match(r"^(?:Теорема|Следствие|Лемма)\s+\d+\.\d+\s+\(([^)]+)\)", block["text"])
        if match:
            names.append({"name": match[1], "block_id": block["block_id"]})
    return names
