from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, TypedDict

from torch.nn import Module

from base import BaseDocument


@dataclass(frozen=True)
class BaseModelType:
    model_name: str


class BaseBatchTokenized(Protocol):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]

    def get_word_ids(self, batch_index: int) -> list[int | None]: ...


class BaseBatchTokenizer(Protocol):
    model_type: BaseModelType

    def tokenize(self, documents: list[BaseDocument]) -> BaseBatchTokenized: ...


class BaseBatchPrediction(TypedDict):
    text_classes: list[list[int]]
    token_tags: list[list[int]]


class BaseBatchModel(ABC):
    model_type: BaseModelType
    # Явно заданное в docs/subword_model.mmd исключение для зависимости Base*.
    encoder: Module

    @abstractmethod
    def forward(self, batch_input: BaseBatchTokenized) -> BaseBatchPrediction: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
