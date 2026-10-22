from abc import ABC, abstractmethod

from .model import BaseBertModelLabeldInput, BaseBertModelOutput


class BaseMetric(ABC):
    @abstractmethod
    def eval(
        self,
        y_true: list[BaseBertModelLabeldInput],
        y_pred: list[BaseBertModelOutput],
    ) -> float: ...

    @abstractmethod
    def get_name(self) -> str: ...


class BaseMetricEvaluator(ABC):
    @abstractmethod
    def add(self, metric: BaseMetric) -> None: ...

    @abstractmethod
    def eval(
        self,
        y_true: list[BaseBertModelLabeldInput],
        y_pred: list[BaseBertModelOutput],
    ) -> dict[str, float]: ...
