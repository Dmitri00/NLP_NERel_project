from abc import ABC, abstractmethod

from .examples import BaseDocument, BaseTextPrediction


class BaseModel(ABC):
    @abstractmethod
    def infer(self, documents: list[BaseDocument]) -> list[BaseTextPrediction]: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
