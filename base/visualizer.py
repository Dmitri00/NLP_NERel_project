from abc import ABC, abstractmethod
from typing import overload

from .examples import BaseDocument, BaseNerelExample, BaseTextPrediction


class BaseVisualizer(ABC):
    @overload
    def visualize(
        self, input_text: BaseNerelExample, prediction: BaseTextPrediction
    ) -> None: ...

    @overload
    def visualize(
        self, input_text: BaseDocument, prediction: BaseTextPrediction
    ) -> None: ...

    @abstractmethod
    def visualize(
        self,
        input_text: BaseNerelExample | BaseDocument,
        prediction: BaseTextPrediction,
    ) -> None: ...
