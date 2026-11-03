import copy
import unittest

from rag.indexing import build_passages, build_sections, fit_search_windows, named_results, normalize_math, notation_labels
from rag.retrieval import SearchIndex


class IndexingContracts(unittest.TestCase):
    def test_named_results_use_corpus_titles_only(self):
        blocks = [
            {"block_id": "a", "text": "Теорема 1.2 (Название). Формулировка"},
            {"block_id": "b", "text": "Доказательство. Применим теорему (Название)."},
        ]
        self.assertEqual(named_results(blocks), [{"name": "Название", "block_id": "a"}])

    def test_long_search_windows_keep_the_tail_and_original_evidence(self):
        def tokenizer(text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}
        source = {"block_id": "a", "text": "Исходный текст", "search_text": "x" * 1100 + "КОНЕЦ"}
        windows = fit_search_windows([source], tokenizer)
        self.assertTrue(all(len(p["search_text"]) <= 480 for p in windows))
        self.assertTrue(windows[-1]["search_text"].endswith("КОНЕЦ"))
        self.assertTrue(all(p["text"] == source["text"] and p["block_id"] == "a" for p in windows))

    def test_section_catalog_contains_results_with_original_ids(self):
        blocks = [
            {"block_id": "a", "section": "S", "text": "Теорема 1.1. Утверждение"},
            {"block_id": "b", "section": "S", "text": "Доказательство. Рассмотрим..."},
            {"block_id": "c", "section": "S", "text": "Следствие 1.1. Утверждение"},
            {"block_id": "d", "section": "T", "text": "Текст"},
        ]
        self.assertEqual(build_sections(blocks), [
            {"title": "S", "result_block_ids": ["a", "c"]},
            {"title": "T", "result_block_ids": []},
        ])

    def test_search_returns_distinct_original_sources(self):
        passages = [
            {"block_id": source, "page": 1, "section": "Теория", "text": text, "search_text": text}
            for source, text in [("a", "интеграл интеграл"), ("a", "интеграл"), ("b", "интеграл Коши")]
        ]
        index = SearchIndex(passages, {"bm25_k1": 1.3, "bm25_b": .7})
        hits = index.search("интеграл", 2)
        self.assertEqual({passages[i]["block_id"] for i, _ in hits}, {"a", "b"})

    def test_formula_aliases_preserve_signs_and_derivative_order(self):
        self.assertEqual(normalize_math(r"\tfrac12(f_x+i f_y)"), r"\frac{1}{2}(f_x+i f_y)")
        self.assertEqual(normalize_math(r"\bar z"), r"\overline{z}")
        self.assertEqual(normalize_math("z\u0304"), r"\overline{z}")
        self.assertNotEqual(normalize_math("f_x+i f_y"), normalize_math("f_x-i f_y"))
        self.assertNotEqual(normalize_math("f'(0)"), normalize_math("f''(0)"))

    def test_notation_labels_are_derived_from_symbols(self):
        self.assertIn("производная", notation_labels("f''(0)=6"))
        self.assertIn("производная", notation_labels(r"f^{(n)}(a)"))
        self.assertIn("экспонента", notation_labels(r"\int_\gamma e^z/z^3"))
        self.assertEqual(notation_labels("Произвольный текст без формул"), [])

    def test_views_and_continuations_preserve_original_evidence(self):
        blocks = [
            {"block_id": "a", "page": 1, "section": "S", "text": "Теорема 2.3. Пусть f непрерывна\n$f(z)=z^2$"},
            {"block_id": "b", "page": 2, "section": "S", "text": "и выполнены условия. Тогда получаем формулу.\n$f'(z)=2z$"},
            {"block_id": "c", "page": 2, "section": "S", "text": "Доказательство. Рассмотрим интеграл.\n$\\int f(z)dz$"},
        ]
        before = copy.deepcopy(blocks)
        views = build_passages(blocks, 1700)
        self.assertEqual(blocks, before)
        original = {block["block_id"]: block for block in blocks}
        for view in views:
            self.assertEqual(view["text"], original[view["block_id"]]["text"])
            self.assertIn(view["block_id"], original)
            expected = ["c"] if view["block_id"] == "c" else ["a", "b"]
            self.assertEqual(view["context_block_ids"], expected)


if __name__ == "__main__":
    unittest.main()
