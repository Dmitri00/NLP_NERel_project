from abc import ABC, abstractmethod
from typing import Self

from .examples import BaseNerelExample


class BaseNerelDataset(ABC):
    """Разбиения исходных документов и интерфейс их хранения."""

    train: list[BaseNerelExample]
    test: list[BaseNerelExample]

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Self: ...
