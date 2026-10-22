from typing import Protocol

from .examples import BaseNerelExample


class BaseTokenized(Protocol):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]

    def get_word_ids(self, batch_index: int) -> list[int | None]: ...


class BaseTokenizer(Protocol):
    def tokenize(self, documents: list[BaseNerelExample]) -> BaseTokenized: ...
