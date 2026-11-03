"""Подсчёт токенов с тем же шаблоном Qwen, который использует сервер."""

from pathlib import Path

import jinja2
from tokenizers import Tokenizer


def raise_template_error(message):
    raise ValueError(message)


class TokenCounter:
    def __init__(self, folder):
        folder = Path(folder)
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_template_error
        self.template = env.from_string((folder / "chat_template.jinja").read_text())

    def render(self, messages, thinking=False):
        return self.template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=thinking,
            tools=None,
        )

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def messages(self, messages, thinking=False):
        return self.count(self.render(messages, thinking))
