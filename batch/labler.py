from dataclasses import dataclass

from base import BaseNerelExample

from .interfaces import BaseBatchTokenized


@dataclass
class BaseLabeledTokenizedBatch(BaseBatchTokenized):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    tags: list[list[int]]
    cls_vec: list[list[int]]
    _tokenized: BaseBatchTokenized

    def get_word_ids(self, batch_index: int) -> list[int | None]:
        return self._tokenized.get_word_ids(batch_index)


class BaseBatchLabler:
    """NER-цель только на первом субтокене; остальные позиции имеют -100."""

    def build_batch(
        self,
        documents: list[BaseNerelExample],
        batch_tokenized: BaseBatchTokenized,
    ) -> BaseLabeledTokenizedBatch:
        tags: list[list[int]] = []
        for batch_index, document in enumerate(documents):
            seen: set[int] = set()
            row: list[int] = []
            for word_id, active in zip(
                batch_tokenized.get_word_ids(batch_index),
                batch_tokenized.attention_mask[batch_index],
            ):
                if active and word_id is not None and word_id not in seen:
                    row.append(document["tags"][word_id].id)
                    seen.add(word_id)
                else:
                    row.append(-100)
            tags.append(row)
        return BaseLabeledTokenizedBatch(
            input_ids=batch_tokenized.input_ids,
            attention_mask=batch_tokenized.attention_mask,
            tags=tags,
            cls_vec=[document["cls_vec"].copy() for document in documents],
            _tokenized=batch_tokenized,
        )
