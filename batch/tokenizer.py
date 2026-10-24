from dataclasses import dataclass
from typing import cast

from transformers import PreTrainedTokenizerFast

from base import BaseDocument

from .interfaces import BaseBatchTokenized, BaseModelType


@dataclass
class _TokenizedBatch:
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    _word_ids: list[list[int | None]]

    def get_word_ids(self, batch_index: int) -> list[int | None]:
        return self._word_ids[batch_index].copy()


class HuggingFaceBatchTokenizer:
    """Одно окно на документ с обрезкой до максимальной длины модели."""

    def __init__(
        self,
        model_type: BaseModelType,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
    ) -> None:
        self.model_type = model_type
        self._tokenizer = tokenizer
        self._max_length = max_length

    def tokenize(self, documents: list[BaseDocument]) -> BaseBatchTokenized:
        encoded = self._tokenizer(
            [document.text for document in documents],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        input_ids = cast(list[list[int]], encoded["input_ids"])
        return _TokenizedBatch(
            input_ids=input_ids,
            attention_mask=cast(list[list[int]], encoded["attention_mask"]),
            _word_ids=[encoded.word_ids(index) for index in range(len(input_ids))],
        )
