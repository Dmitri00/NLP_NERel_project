from abc import ABC, abstractmethod
from typing import Self

from .examples import TextTag


class BaseTextTagFabric(ABC):
    @abstractmethod
    def fit(self, tags: list[str]) -> None: ...

    @abstractmethod
    def by_id(self, id: int) -> TextTag: ...

    @abstractmethod
    def by_value(self, value: str) -> TextTag: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Self: ...
