import copy
import json
import unittest
from pathlib import Path

from rag.budget import TokenCounter
from rag.common import check_answer, check_questions
from rag.pdf_extract import Node, inline
from scripts.evaluate import citation_f1, score

Q = {
    "id": "arbitrary-id",
    "question": "Which?",
    "options": [{"id": s, "text": s} for s in ["alpha", "two", "C", "99"]],
}
A = {
    "id": "arbitrary-id",
    "answer_ids": ["two"],
    "abstain": False,
    "rationale": "Evidence.",
    "citations": ["source-X"],
}


class Contracts(unittest.TestCase):
    def test_terminal_period_does_not_allow_multiple_answers(self):
        from predict import parse_json

        self.assertEqual(parse_json(json.dumps(A) + "."), A)
        for suffix in [". " + json.dumps(A), " trailing explanation", ".."]:
            with self.assertRaises(json.JSONDecodeError):
                parse_json(json.dumps(A) + suffix)

    def test_context_budget_and_request_use_the_selected_system(self):
        from predict import pack_context

        calls = []
        class Counter:
            def count(self, text):
                return 1

            def messages(self, messages, thinking):
                calls.append(messages)
                return 2

        source = {"block_id": "source-X", "page": 1, "section": "S", "text": "Evidence"}
        messages, selected = pack_context(
            Q, [source], Counter(),
            {"retrieval": {"context_tokens": 10}, "max_output_tokens": 10},
            {"enable_thinking": False, "limits": {"request_tokens": 100}},
            system="Unchanged baseline instruction",
        )
        self.assertEqual(selected, [source])
        self.assertEqual(messages[0]["content"], "Unchanged baseline instruction")
        self.assertEqual(calls[0][0], messages[0])

    def test_latex_json_escape(self):
        from predict import parse_json

        value = parse_json(r'{"rationale":"Формула: \frac{n!}{2\pi i}"}')
        self.assertEqual(value["rationale"], r"Формула: \frac{n!}{2\pi i}")
        self.assertEqual(
            parse_json(r'{"rationale":"line1\nline2"}')["rationale"], "line1\nline2"
        )
        escaped = json.dumps({"rationale": r"\partial f"})
        self.assertEqual(parse_json(escaped)["rationale"], r"\partial f")

    def test_latex_spacing_and_set_difference(self):
        from collections import Counter

        from predict import parse_json, render_question
        from rag.pdf_extract import decode_char

        self.assertEqual(
            parse_json(r'{"rationale":"\quad x\,dx"}')["rationale"], r"\quad x\,dx"
        )
        self.assertEqual(decode_char("\\", "CMSY10", Counter()), r"\setminus ")
        q = {**Q, "question": r"\tfrac12(f_x+i f_y)=0"}
        self.assertIn(r"\frac{1}{2}", render_question(q))

    def test_long_rationale_preserves_conclusion(self):
        from predict import shorten_rationale

        text = (
            "Теорема 4.4 даёт полный дифференциал. "
            + "Промежуточное пояснение. " * 70
            + "Верны beta и gamma."
        )
        out = shorten_rationale(text)
        self.assertLessEqual(len(out), 1200)
        self.assertTrue(out.startswith("Теорема 4.4 даёт полный дифференциал."))
        self.assertTrue(out.endswith("Верны beta и gamma."))

    def test_explicit_abstention_format(self):
        from predict import normalize_answer

        a = {**A, "abstain": True, "answer_ids": []}
        fixed = normalize_answer(a)
        self.assertEqual(fixed["citations"], [])
        check_answer(fixed, Q, {"source-X"})
        conflicting = {**A, "abstain": True}
        with self.assertRaises(ValueError):
            check_answer(normalize_answer(conflicting), Q, {"source-X"})

    def test_actual_option_ids(self):
        check_questions([Q])
        check_answer(A, Q, {"source-X"})
        b = copy.deepcopy(A)
        b["answer_ids"] = ["B"]
        with self.assertRaises(ValueError):
            check_answer(b, Q, {"source-X"})

    def test_abstention_and_citations(self):
        b = copy.deepcopy(A)
        b["abstain"] = True
        with self.assertRaises(ValueError):
            check_answer(b, Q, {"source-X"})
        b["answer_ids"] = []
        b["citations"] = []
        check_answer(b, Q, {"source-X"})
        with self.assertRaises(ValueError):
            check_answer(A, Q, {"different-source"})

    def test_duplicates(self):
        with self.assertRaises(ValueError):
            check_questions([Q, Q])
        b = copy.deepcopy(A)
        b["citations"] *= 2
        with self.assertRaises(ValueError):
            check_answer(b, Q, {"source-X"})

    def test_best_citation_set(self):
        self.assertEqual(citation_f1(["a"], [["b"], ["a"]]), 1)
        self.assertAlmostEqual(citation_f1(["a", "extra"], [["a"]]), 2 / 3)

    def test_scripts_preserve_order(self):
        self.assertEqual(
            inline([Node("x", 0, 5, 10, 10, True), Node("2", 5, 8, 6, 7, True)], 10),
            "x^{2}",
        )
        self.assertEqual(
            inline([Node("z", 0, 5, 10, 10, True), Node("0", 5, 8, 11.5, 7, True)], 10),
            "z_{0}",
        )

    def test_macro_scoring_and_unverifiable_usage(self):
        cats = ["условия", "сопоставление", "применение", "недостаточность данных"]
        qs = []
        gs = []
        ans = []
        ts = []
        for i, c in enumerate(cats):
            q = copy.deepcopy(Q)
            q["id"] = str(i)
            qs.append(q)
            g = {
                "id": str(i),
                "category": c,
                "abstain": False,
                "answer_ids": ["two"],
                "citation_sets": [["source-X"]],
            }
            gs.append(g)
            a = copy.deepcopy(A)
            a["id"] = str(i)
            ans.append(a)
            ts.append(
                {
                    "id": str(i),
                    "llm_calls": 1,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "elapsed_ms": 10,
                    "usage_verified": True,
                    "requests": [
                        {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "reserved_output_tokens": 100,
                        }
                    ],
                }
            )
        r = score(qs, gs, ans, ts, [{"block_id": "source-X"}], c_ref=436)
        self.assertEqual(r["Q"], 1)
        self.assertEqual(r["grade"], 100)
        ts[0]["usage_verified"] = False
        r = score(qs, gs, ans, ts, [{"block_id": "source-X"}], c_ref=436)
        self.assertFalse(r["efficiency_valid"])
        self.assertEqual(r["E"], 0)
        self.assertEqual(r["Q"], 0.75)

    def test_no_context_has_no_Q(self):
        a = copy.deepcopy(A)
        a["citations"] = []
        check_answer(a, Q, set(), no_context=True)

    @unittest.skipUnless(
        Path("local/assets/tokenizer/tokenizer.json").exists(),
        "download tokenizer first",
    )
    def test_budget_uses_same_template_as_server(self):
        from transformers import AutoTokenizer

        messages = [
            {"role": "system", "content": "JSON."},
            {"role": "user", "content": "Производная ∂f/∂z?"},
        ]
        t = AutoTokenizer.from_pretrained(
            "local/assets/tokenizer", local_files_only=True
        )
        expected = len(
            t.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        )
        self.assertEqual(
            TokenCounter("local/assets/tokenizer").messages(messages), expected
        )


if __name__ == "__main__":
    unittest.main()
