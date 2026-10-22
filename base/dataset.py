from abc import ABC, abstractmethod
from typing import Self

from .examples import BaseTokenizedNerelExample


class BaseNerelDataset(list[BaseTokenizedNerelExample], ABC):
    """Список документов и интерфейс хранения без библиотечных зависимостей."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Self: ...
