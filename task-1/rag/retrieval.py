"""Поиск BM25 и E5, объединение результатов и повторное ранжирование."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import snowballstemmer
from rag.indexing import INDEX_VERSION, normalize_math, search_text
from rag.formulas import formula_sources

STEM = snowballstemmer.stemmer("russian")
STOP = set(
    "и в во на к с со о об от из по у для что это то не ни а но или как все всех при пусть если чем какие какой которое которые которой только любой любых верно верны утверждение утверждения функция функции f g z x y n".split()
)
SYMBOLS = {
    "∂": r"\partial",
    "∫": r"\int",
    "∑": r"\sum",
    "π": r"\pi",
    "∞": r"\infty",
    "γ": r"\gamma",
    "ζ": r"\zeta",
    "−": "-",
    "′": "'",
    "≤": r"\le",
    "≥": r"\ge",
    "≡": r"\equiv",
}


def terms(text, normalized=True):
    text = (normalize_math(text) if normalized else text).lower().replace("ё", "е")
    for symbol, replacement in SYMBOLS.items():
        text = text.replace(symbol, replacement)
    words = re.findall(r"[а-я]+|[a-z]+|\d+(?:\.\d+)+", text)
    tokens = []
    for word in words:
        if word not in STOP and len(word) > 1:
            tokens.append(STEM.stemWord(word) if re.fullmatch("[а-я]+", word) else word)
    # Сохраняем обозначения из формул как отдельные поисковые признаки.
    for expr in re.findall(r"\$([^$]+)\$", text):
        compact = re.sub(r"[\s{}]", "", expr)
        tokens.extend(
            "math:" + m
            for m in re.findall(
                r"[a-z](?:[_^][a-z0-9]+|'+)|[0-9]+!|\\(?:int|sum|frac|partial|bar|overline)",
                compact,
            )
        )
    return tokens


def split_block(block, size):
    """Разбить длинный блок по строкам, сохранив его ID для ссылок."""
    parts = []
    lines = []
    total = 0
    for line in block["text"].splitlines():
        if total + len(line) > size and lines:
            parts.append("\n".join(lines))
            lines = []
            total = 0
        lines.append(line)
        total += len(line) + 1
    if lines:
        parts.append("\n".join(lines))
    return [{**block, "text": text, "chunk": i} for i, text in enumerate(parts)]


class SearchIndex:
    def __init__(self, passages, config, normalized=True):
        self.normalized = normalized
        self.encoder = None
        self.vectors = None
        self.reranker = None
        self.passages = passages
        self.config = config
        self.postings = defaultdict(list)
        self.lengths = []
        for i, passage in enumerate(passages):
            tokens = terms(passage.get("search_text", passage["section"] + " " + passage["text"]), normalized)
            self.lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings[term].append((i, count))
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        self.by_id = defaultdict(list)
        for i, passage in enumerate(passages):
            self.by_id[passage["block_id"]].append(i)

    @classmethod
    def load(cls, path):
        data = json.loads((Path(path) / "index.json").read_text())
        if data.get("version") != INDEX_VERSION:
            raise ValueError("Index format changed; rebuild it with build_index.py")
        result = cls(data["passages"], data["retrieval"])
        result.source_blocks = {block["block_id"]: block for block in data["blocks"]}
        result.source_order = list(result.source_blocks)
        result.source_positions = {block_id: i for i, block_id in enumerate(result.source_order)}
        result.sections = data["sections"]
        result.named_results = data["named_results"]
        result.formula_index = data["formula_index"]
        result.section_indexes = {
            section["title"]: cls([p for p in data["passages"] if p["section"] == section["title"]], data["retrieval"])
            for section in result.sections
        }
        result.result_indexes = {
            section["title"]: cls([
                p for p in data["passages"] if p["block_id"] in section["result_block_ids"]
            ], data["retrieval"])
            for section in result.sections
        }
        result.source_groups = {
            p["block_id"]: p["context_block_ids"] for p in data["passages"]
        }
        result.baseline = cls(
            [part for block in data["blocks"] for part in split_block(block, data["retrieval"]["chunk_chars"])],
            data["retrieval"], normalized=False,
        )
        result.baseline.source_blocks = result.source_blocks
        import numpy as np

        result.vectors = np.load(Path(path) / "vectors.npy", allow_pickle=False)
        result.section_vectors = np.load(Path(path) / "section_vectors.npy", allow_pickle=False)
        result.embedding_path = Path(path) / "embedding"
        result.reranker_path = Path(path) / "reranker"
        return result

    def search(self, query, k=35):
        scores = defaultdict(float)
        document_count = len(self.passages)
        k1 = self.config["bm25_k1"]
        b = self.config["bm25_b"]
        for term, query_frequency in Counter(terms(query, self.normalized)).items():
            hits = self.postings.get(term, [])
            idf = math.log(1 + (document_count - len(hits) + 0.5) / (len(hits) + 0.5))
            for i, term_frequency in hits:
                scores[i] += (
                    idf
                    * term_frequency
                    * (k1 + 1)
                    / (
                        term_frequency
                        + k1 * (1 - b + b * self.lengths[i] / max(self.avg, 1))
                    )
                    * min(query_frequency, 2)
                )
        ranked = []
        seen = set()
        for i, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            block_id = self.passages[i]["block_id"]
            if block_id not in seen:
                ranked.append((i, score))
                seen.add(block_id)
                if len(ranked) == k:
                    break
        return ranked

    def retrieve(self, question, mode):
        if mode == "baseline" and self.normalized:
            return self.baseline.retrieve(question, mode)
        settings = self.config
        full = (
            question["question"]
            + "\n"
            + "\n".join(option["text"] for option in question["options"])
        )
        if self.normalized:
            full = search_text(full)
        queries = [full]
        if mode != "baseline" and settings["multiquery"]:
            queries += [search_text(question["question"])] + [
                search_text(option["text"]) for option in question["options"]
            ]
        rankings = [self.search(query, settings["pool_size"]) for query in queries]
        section = None
        if mode == "no_dense" and rankings[0]:
            title = self.passages[rankings[0][0][0]]["section"]
            section = next(item for item in self.sections if item["title"] == title)
        fused = defaultdict(float)
        # RRF объединяет места в выдаче: шкалы BM25 и E5 напрямую несопоставимы.
        for query_index, ranking in enumerate(rankings):
            for rank, (i, _) in enumerate(ranking, 1):
                fused[i] += (1.5 if query_index == 0 else 1) / (35 + rank)
        if mode not in ("baseline", "no_dense"):
            import numpy as np

            from rag.dense import Encoder

            if self.encoder is None:
                self.encoder = Encoder(self.embedding_path)
            vectors = self.encoder.encode(
                [full, search_text(question["question"])], "query"
            )
            section = self.sections[int(np.argmax(self.section_vectors @ vectors[0]))]
            for vector in vectors:
                scores = self.vectors @ vector
                seen = set()
                for i in np.argsort(-scores):
                    block_id = self.passages[i]["block_id"]
                    if block_id in seen:
                        continue
                    seen.add(block_id)
                    fused[int(i)] += 4 / (35 + len(seen))
                    if len(seen) == settings["pool_size"]:
                        break
        # Складываем свидетельства разных поисковых представлений одного блока,
        # но показываем reranker только его наиболее подходящее окно.
        source_scores = defaultdict(float)
        best_view = {}
        for i, value in fused.items():
            block_id = self.passages[i]["block_id"]
            source_scores[block_id] += value
            if block_id not in best_view or value > fused[best_view[block_id]]:
                best_view[block_id] = i
        fused = {best_view[block_id]: value for block_id, value in source_scores.items()}
        for i in fused:
            if mode != "baseline" and re.match(
                r"^(Теорема|Определение|Следствие|Лемма|Замечание)\s",
                self.passages[i]["text"],
            ):
                fused[i] *= 1.35
        ranked = sorted(fused, key=lambda i: (-fused[i], i))
        # Несколько поисковых окон одного источника не должны занять весь top-k.
        unique, source_ids = [], set()
        for i in ranked:
            block_id = self.passages[i]["block_id"]
            if block_id not in source_ids:
                unique.append(i)
                source_ids.add(block_id)
        ranked = unique
        candidate_ids = [self.passages[i]["block_id"] for i in ranked[:settings["rerank_pool_size"]]]
        rerank_scores = {}
        if mode not in ("baseline", "no_reranker"):
            from rag.rerank import Reranker

            if self.reranker is None:
                self.reranker = Reranker(self.reranker_path)
            pool = ranked[: settings["rerank_pool_size"]]
            values = self.reranker.score(
                full,
                [
                    self.passages[i].get("search_text", self.passages[i]["section"] + "\n" + self.passages[i]["text"])
                    for i in pool
                ],
            )
            rerank_scores = dict(zip(pool, values))
            ranked = (
                sorted(pool, key=lambda i: (-rerank_scores[i], -fused[i], i))
                + ranked[len(pool) :]
            )
        top = ranked[: 4 if mode == "baseline" else settings["top_k"]]
        top_ids = [self.passages[i]["block_id"] for i in top]
        # Раздел добавляет вес только кандидатам глобального поиска.
        # Обязательных мест для фрагментов раздела нет.
        section_ids = []
        if section is not None:
            local = self.section_indexes[section["title"]]
            results = self.result_indexes[section["title"]]
            global_ids = [self.passages[i]["block_id"] for i in ranked]
            scores = {block_id: 1 / (35 + rank) for rank, block_id in enumerate(global_ids, 1)}
            pool_ids = set(candidate_ids)
            for local_index in (local, results):
                for rank, (i, _) in enumerate(local_index.search(full, settings["pool_size"]), 1):
                    block_id = local_index.passages[i]["block_id"]
                    if block_id in pool_ids:
                        scores[block_id] += 0.5 / (35 + rank)
                        section_ids.append(block_id)
            supported = sorted(global_ids, key=lambda block_id: -scores[block_id])
            top_ids = list(dict.fromkeys(top_ids[:2] + supported))[:settings["top_k"]]
        formula_ids = []
        if mode != "baseline":
            query_terms = set(terms(question["question"]))
            named_ids = [
                item["block_id"] for item in self.named_results
                if terms(item["name"]) and set(terms(item["name"])) <= query_terms
            ]
            formula_ids = formula_sources(question["question"], self.formula_index)
            top_ids = list(dict.fromkeys(named_ids + formula_ids + top_ids))[:settings["top_k"]]
            section_ids = named_ids + section_ids
        candidates, seen, extra = [], set(), []
        groups = getattr(self, "source_groups", {})
        for block_id in top_ids:
            for member in groups.get(block_id, [block_id]):
                if member not in seen:
                    candidates.append({
                        **self.source_blocks[member],
                        "retrieval_score": source_scores.get(block_id, 0), "neighbor": False,
                    })
                    seen.add(member)
            # Дополнительные соседи следуют после основных результатов.
            if mode != "baseline" and settings["neighbors"]:
                passage = self.source_blocks[block_id]
                position = self.source_positions[block_id]
                for j in (position - 1, position + 1):
                    if 0 <= j < len(self.source_order):
                        neighbor = self.source_blocks[self.source_order[j]]
                        if neighbor["section"] == passage["section"] and abs(neighbor["page"] - passage["page"]) <= 1:
                            extra.append(neighbor)
        for passage in extra:
            if passage["block_id"] not in seen:
                candidates.append({**passage, "retrieval_score": 0, "neighbor": True})
                seen.add(passage["block_id"])
        return candidates, {
            "queries": queries,
            "candidate_block_ids": list(dict.fromkeys(candidate_ids + section_ids + formula_ids)),
            "section": section["title"] if section else None,
            "section_block_ids": list(dict.fromkeys(section_ids)),
            "formula_block_ids": formula_ids,
            "ranked_block_ids": [
                self.passages[i]["block_id"] for i in ranked[: settings["pool_size"]]
            ],
            "reranker_scores": {
                self.passages[i]["block_id"]: score
                for i, score in rerank_scores.items()
                if self.passages[i]["block_id"] in top_ids
            },
            "strategy": "hybrid search, reranking, single LLM call",
        }
