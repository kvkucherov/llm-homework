"""Извлечение текста и формул из PDF учебника.

Правила привязаны к шрифтам и вёрстке этого PDF. Нераспознанные символы
попадают в журнал, чтобы их можно было сверить с оригиналом."""

import re
from collections import Counter
from dataclasses import dataclass

LATEX = dict(
    zip(
        "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ",
        [
            "\\" + x + " "
            for x in "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega".split()
        ],
    )
)
LATEX.update(
    {
        "∂": r"\partial ",
        "∞": r"\infty ",
        "∈": r"\in ",
        "∉": r"\notin ",
        "⊂": r"\subset ",
        "⊆": r"\subseteq ",
        "≡": r"\equiv ",
        "≤": r"\le ",
        "≥": r"\ge ",
        "→": r"\to ",
        "⇒": r"\Rightarrow ",
        "↔": r"\leftrightarrow ",
        "⇔": r"\Leftrightarrow ",
        "−": "-",
        "·": r"\cdot ",
        "×": r"\times ",
        "∇": r"\nabla ",
        "∅": r"\emptyset ",
        "∪": r"\cup ",
        "∩": r"\cap ",
        "±": r"\pm ",
        "̸": r"\not ",
        "′": "'",
        "∥": r"\Vert ",
        "√": r"\sqrt ",
        "□": "",
    }
)
LATEX.update({"\\": r"\setminus ", "{": r"\{", "}": r"\}"})
# В CMEX10 буква Z обозначает интеграл: обычное извлечение Unicode это теряет.
EX = {
    "Z": r"\int ",
    "R": r"\int ",
    "X": r"\sum ",
    "P": r"\sum ",
    "Y": r"\prod ",
    "Q": r"\prod ",
    "p": r"\sqrt ",
    "q": r"\sqrt ",
    "r": r"\sqrt ",
    "s": r"\sqrt ",
    "\x12": "(",
    "\x13": ")",
    "\x10": "(",
    "\x11": ")",
    "\x00": "(",
    "\x01": ")",
    "\x02": "[",
    "\x03": "]",
    "\x14": "[",
    "\x15": "]",
    "\x08": "{",
    "\x09": "}",
    "\x1a": "{",
    "\x1b": "}",
    "\x0c": "|",
    "\uf8eb": "(",
    "\uf8ed": "",
    "\uf8ec": "",
    "\uf8f6": ")",
    "\uf8f7": "",
    "\uf8f8": "",
    "\uf8f1": "{",
    "\uf8f2": "",
    "\uf8f3": "",
    "\uf8f4": "",
    "\uf8fc": "}",
    "\uf8fd": "",
    "\uf8fe": "",
}


@dataclass
class Node:
    text: str
    x: float
    x1: float
    y: float
    size: float
    math: bool
    font: str = ""

    @property
    def cx(self):
        return (self.x + self.x1) / 2


def decode_char(char, font, unknown):
    if font.startswith("F") and any(0xC0 <= ord(x) <= 0xFF for x in char):
        return char.encode("latin1").decode("cp1251")
    if font.startswith("F"):
        return {"\x02": "-", "\x15": "–", "\x16": "—", "\x1f": "fi", "\x9f": "§"}.get(
            char, char
        )
    if font.startswith("CMEX"):
        if char in EX:
            return EX[char]
        if char.strip() and (ord(char) < 32 or ord(char) > 0xF000):
            unknown[f"{font}:{ord(char):04x}"] += 1
            return "[неразобранный символ]"
    if font.startswith("MSBM") and char.isalpha():
        return r"\mathbb{" + char + "}"
    if ord(char) < 32 and char.strip():
        unknown[f"{font}:{ord(char):04x}"] += 1
    return LATEX.get(char, char)


def inline(nodes, baseline=None):
    if not nodes:
        return ""
    if baseline is None:
        weights = Counter()
        for node in nodes:
            weights[round(node.y, 1)] += max(node.x1 - node.x, 1) * (
                2 if node.size > 8 else 0.4
            )
        baseline = weights.most_common(1)[0][0]
    parts = []
    previous = None
    active_script = None
    base_size = max(node.size for node in nodes)
    for node in sorted(nodes, key=lambda node: (round(node.x, 1), node.y)):
        shift = (
            "^"
            if node.y < baseline - 1.6 and node.size < base_size * 0.9
            else "_"
            if node.y > baseline + 0.7 and node.size < base_size * 0.9
            else None
        )
        if node.text == "'":
            shift = None
        if shift != active_script:
            if active_script:
                parts.append("}")
            if shift:
                parts.append(shift + "{")
            active_script = shift
        if (
            previous
            and not active_script
            and node.x - previous.x1 > max(1.4, node.size * 0.2)
        ):
            parts.append(" ")
        parts.append(node.text)
        previous = node
    if active_script:
        parts.append("}")
    return "".join(parts)


def extract_page(page):
    unknown = Counter()
    nodes = []
    for b in page.get_text("rawdict")["blocks"]:
        for line in b.get("lines", []):
            for span in line["spans"]:
                for char in span["chars"]:
                    x, y = char["origin"]
                    if y < 44 or not char["c"].strip():
                        continue
                    font = span["font"]
                    value = decode_char(char["c"], font, unknown)
                    if not value:
                        continue
                    if (
                        font.startswith("CMEX")
                        and char["c"] in "ZXYPQR\x12\x13\x10\x11"
                    ):
                        y += span["size"] * (
                            0.75
                            if char["c"] in "PQ"
                            else 0.95
                            if char["c"] in "XY"
                            else 1.36
                        )
                    nodes.append(
                        Node(
                            value,
                            x,
                            char["bbox"][2],
                            y,
                            span["size"],
                            not font.startswith("F"),
                            font,
                        )
                    )
    bars = []
    for drawing in page.get_drawings():
        if (drawing.get("width") or 2) > 1:
            continue
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1:]
                if abs(a.y - b.y) < 0.15 and 3 < abs(a.x - b.x) < 160:
                    bars.append((min(a.x, b.x), max(a.x, b.x), a.y))
    fractions = 0
    overlines = 0
    for x, x1, y in sorted(bars, key=lambda b: b[1] - b[0]):
        above = [
            node
            for node in nodes
            if node.math and x - 0.7 <= node.cx <= x1 + 0.7 and -16 < node.y - y < -0.7
        ]
        below = [
            node
            for node in nodes
            if node.math and x - 0.7 <= node.cx <= x1 + 0.7 and 3 < node.y - y < 18
        ]
        if (
            above
            and below
            and min(
                node.y - y for node in below if node.size == max(t.size for t in below)
            )
            > 6
        ):
            # Линия, пересекающая обычный текст, скорее всего не является чертой дроби.
            if any(
                not node.math and x <= node.cx <= x1 and abs(node.y - y) < 12
                for node in nodes
            ):
                continue
            value = r"\frac{" + inline(above) + "}{" + inline(below) + "}"
            used = {id(node) for node in above + below}
            nodes = [node for node in nodes if id(node) not in used]
            nodes.append(Node(value, x, x1, y + 2.5, 10, True, "fraction"))
            fractions += 1
        elif below and x1 - x < 30:
            near = [node for node in below if 3 < node.y - y < 10]
            if near:
                used = {id(node) for node in near}
                nodes = [node for node in nodes if id(node) not in used]
                nodes.append(
                    Node(
                        r"\overline{" + inline(near) + "}",
                        x,
                        x1,
                        max(node.y for node in near),
                        10,
                        True,
                        "bar",
                    )
                )
                overlines += 1
    # Сначала присоединяем пределы к операторам, иначе они попадут в соседние строки.
    for operator in [
        node for node in nodes if node.text.strip() in (r"\int", r"\sum", r"\prod")
    ]:
        limits = [
            node
            for node in nodes
            if node is not operator
            and node.size < 9
            and operator.x - 2 <= node.cx <= operator.x1 + 6
            and abs(node.y - operator.y) < 17
        ]
        for high in (True, False):
            side = [
                node
                for node in limits
                if (node.y < operator.y - 3 if high else node.y > operator.y + 5)
            ]
            if side:
                operator.text = (
                    operator.text.strip()
                    + ("^{" if high else "_{")
                    + inline(side)
                    + "} "
                )
                used = {id(node) for node in side}
                nodes = [node for node in nodes if id(node) not in used]
    anchors = []
    for node in sorted(
        (node for node in nodes if node.size >= 9), key=lambda node: node.y
    ):
        if not anchors or node.y - anchors[-1] > 1.8:
            anchors.append(node.y)
    if not anchors:
        anchors = sorted({node.y for node in nodes})
    groups = {y: [] for y in anchors}
    for node in nodes:
        y = min(anchors, key=lambda y: abs(y - node.y))
        groups[y].append(node)
    rows = []
    for y, line_nodes in sorted(groups.items()):
        if not line_nodes:
            continue
        line_nodes.sort(key=lambda node: node.x)
        segments = []
        group = []
        is_math = line_nodes[0].math
        for node in line_nodes:
            if node.math != is_math:
                segments.append((is_math, inline(group, y)))
                group = []
                is_math = node.math
            group.append(node)
        segments.append((is_math, inline(group, y)))
        text = " ".join(
            "$" + t.strip() + "$" if math else t for math, t in segments if t.strip()
        )
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        rows.append({"text": text, "x": min(node.x for node in line_nodes), "y": y})
    return rows, {
        "fractions": fractions,
        "overlines": overlines,
        "unsupported": dict(unknown),
    }


HEADING = re.compile(
    r"^(?:Теорема|Определение|Следствие|Лемма|Замечание|Пример)\s+\d+\.\d+(?:\.|\s*\()"
)


def page_blocks(rows, page, section, max_chars=1700):
    chunks = []
    current = []

    def flush():
        nonlocal current
        if current:
            text = "\n".join(current)
            text = re.sub(r"([а-яё])-\n([а-яё])", r"\1\2", text)
            chunks.append(
                {
                    "block_id": f"b_p{page:04d}_{len(chunks) + 1:02d}",
                    "page": page,
                    "section": section,
                    "text": text,
                }
            )
            current = []

    for row in rows:
        text = row["text"]
        if re.match(r"^§\s*\d+", text):
            flush()
            section = text
        if HEADING.match(text) or text.startswith("Доказательство."):
            flush()
        if sum(len(s) + 1 for s in current) + len(text) > max_chars:
            flush()
        current.append(text)
    flush()
    return chunks
