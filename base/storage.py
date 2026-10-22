from abc import ABC, abstractmethod

from .dataset import BaseNerelDataset


class BaseNerelDatasetBundle(ABC):
    @abstractmethod
    def label2id(self, label: str) -> int: ...

    @abstractmethod
    def id2label(self, id: int) -> str: ...

    @abstractmethod
    def dataset(self) -> BaseNerelDataset: ...


class BaseDatasetStorage(ABC):
    @abstractmethod
    def load(self, path: str) -> BaseNerelDatasetBundle: ...

    @abstractmethod
    def save(self, path: str, dataset: BaseNerelDatasetBundle) -> None: ...
