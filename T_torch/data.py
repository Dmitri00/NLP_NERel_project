from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self, TypedDict, cast

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase

from various import BaseBertModelLabeldInput

IGNORE_LABEL = -100


class ModelInputTensors(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dataclass
class TTorchModelInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device) -> Self:
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self

    def to_dict(self) -> ModelInputTensors:
        return ModelInputTensors(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        )


class TTorchBatchCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=IGNORE_LABEL,
            return_tensors="pt",
        )

    def collate(
        self, features: Sequence[BaseBertModelLabeldInput]
    ) -> tuple[TTorchModelInput, torch.Tensor, torch.Tensor]:
        # BIO-метки дополняются с той же стороны, что и входные токены.
        batch = self._collator(
            [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                    "labels": feature["labels"],
                }
                for feature in features
            ]
        )
        inputs = TTorchModelInput(
            input_ids=cast(torch.Tensor, batch["input_ids"]),
            attention_mask=cast(torch.Tensor, batch["attention_mask"]),
        )
        labels = cast(torch.Tensor, batch["labels"]).masked_fill(
            ~inputs.attention_mask.bool(), IGNORE_LABEL
        )
        document_labels = torch.tensor(
            [feature["document_labels"] for feature in features], dtype=torch.float32
        )
        return inputs, labels, document_labels


class _NerelTorchDataset(Dataset[BaseBertModelLabeldInput]):
    """Размеченные входы модели с интерфейсом Dataset в PyTorch."""

    def __init__(self, dataset: Sequence[BaseBertModelLabeldInput]) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> BaseBertModelLabeldInput:
        return self._dataset[index]
