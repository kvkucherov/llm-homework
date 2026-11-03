"""Оценка пар «вопрос — фрагмент» моделью mMiniLM на CPU."""

import os


class Reranker:
    def __init__(self, path):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False
            )
            .to("cpu")
            .eval()
        )
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.parameters > 1_000_000_000:
            raise ValueError("Reranker exceeds parameter limit")

    def score(self, query, texts, batch_size=4):
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                group = texts[start : start + batch_size]
                inputs = self.tokenizer(
                    [query] * len(group),
                    group,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                scores.extend(self.model(**inputs).logits.flatten().tolist())
        return scores
