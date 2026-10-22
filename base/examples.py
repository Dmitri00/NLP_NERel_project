from typing import TypedDict


class BaseNerelExample(TypedDict):
    tokens: list[str]
    tags: list[str]
    cls_vec: list[int]


class BaseTokenizedNerelExample(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    cls_vec: list[int]
