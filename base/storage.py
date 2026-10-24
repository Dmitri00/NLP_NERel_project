from abc import ABC, abstractmethod

from .dataset import BaseNerelDataset
from .tags import BaseTextTagFabric


class BaseNerelDatasetBundle(ABC):
    tag_fabric: BaseTextTagFabric

    @abstractmethod
    def dataset(self) -> BaseNerelDataset: ...


class BaseDatasetStorage(ABC):
    @abstractmethod
    def load(self, path: str) -> BaseNerelDatasetBundle: ...

    @abstractmethod
    def save(self, path: str, dataset: BaseNerelDatasetBundle) -> None: ...
