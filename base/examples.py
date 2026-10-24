from dataclasses import dataclass
from typing import TypedDict

TextToken = str
TextTopic = int


@dataclass
class BaseDocument:
    text: list[TextToken]


@dataclass(frozen=True)
class TextTag:
    id: int
    value: str


class BaseNerelExample(TypedDict):
    tokens: BaseDocument
    tags: list[TextTag]
    cls_vec: list[TextTopic]


class BaseTextPrediction(TypedDict):
    input_text: BaseDocument
    tag_prediction: list[TextTag]
    topic_prediction: list[TextTopic]
