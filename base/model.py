from abc import ABC, abstractmethod
from typing import TypedDict, overload


class BaseBertModelInput(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]


class BaseBertModelLabeldInput(BaseBertModelInput):
    labels: list[int]
    document_labels: list[int]


class BaseBertModelOutput(TypedDict):
    text_classes: list[int]
    token_tags: list[int]


class BaseModel(ABC):
    @overload
    def infer(
        self, batch_input: list[BaseBertModelInput]
    ) -> list[BaseBertModelOutput]: ...

    @overload
    def infer(
        self, batch_input: list[BaseBertModelLabeldInput]
    ) -> list[BaseBertModelOutput]: ...

    @abstractmethod
    def infer(
        self,
        batch_input: list[BaseBertModelInput] | list[BaseBertModelLabeldInput],
    ) -> list[BaseBertModelOutput]: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
