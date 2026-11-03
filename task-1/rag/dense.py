"""Векторные представления E5. Модель загружается с диска и работает на CPU."""

import os

import numpy as np


class Encoder:
    def __init__(self, path):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModel.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False
            )
            .to("cpu")
            .eval()
        )
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.parameters > 1_000_000_000:
            raise ValueError("Embedding model exceeds assignment parameter limit")

    def encode(self, texts, prefix, batch_size=8):
        vectors = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    [prefix + ": " + text for text in texts[i : i + batch_size]],
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                hidden_states = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                # Усредняем только настоящие токены, без padding.
                embeddings = (hidden_states * mask).sum(1) / mask.sum(1)
                embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
                vectors.append(embeddings.numpy())
        return np.concatenate(vectors).astype(np.float32)
