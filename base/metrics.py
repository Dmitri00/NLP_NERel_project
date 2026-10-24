from abc import ABC, abstractmethod

from .examples import BaseNerelExample, BaseTextPrediction


class BaseMetric(ABC):
    """Для BIO-метрик O исключается из оцениваемых классов.

    Перед расчётом из ground truth и предсказаний синхронно удаляются
    только пары (O, O). Пары (O, сущность) и (сущность, O) сохраняются
    как ошибки. При пустом остатке результат 0.
    """

    @abstractmethod
    def eval(
        self,
        y_true: list[BaseNerelExample],
        y_pred: list[BaseTextPrediction],
    ) -> float: ...

    @abstractmethod
    def get_name(self) -> str: ...


class BaseMetricEvaluator(ABC):
    @abstractmethod
    def add(self, metric: BaseMetric) -> None: ...

    @abstractmethod
    def eval(
        self,
        y_true: list[BaseNerelExample],
        y_pred: list[BaseTextPrediction],
    ) -> dict[str, float]: ...
