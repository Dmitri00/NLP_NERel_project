from dataclasses import dataclass
from pathlib import Path
from typing import cast, overload

import torch
from torch import nn
from transformers import PreTrainedModel

from base import (
    BaseBertModelInput,
    BaseBertModelLabeldInput,
    BaseBertModelOutput,
    BaseModel,
)

IGNORE_LABEL = -100


@dataclass
class TTorchJointOutput:
    logits: torch.Tensor
    cls_logits: torch.Tensor


class JointNERAndTopicModel(nn.Module):
    """Общий backbone с головами классификации токенов и документов."""

    def __init__(
        self,
        encoder: PreTrainedModel,
        num_token_tags: int,
        num_document_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # AutoModel может вернуть полную encoder-decoder модель: нужен её encoder.
        self._encoder = (
            cast(PreTrainedModel, encoder.get_encoder())
            if encoder.config.is_encoder_decoder
            else encoder
        )
        self._dropout = nn.Dropout(dropout)
        # Конфиги Transformers отображают свои имена размера на hidden_size.
        self._token_head = nn.Linear(
            int(self._encoder.config.hidden_size), num_token_tags
        )
        self._cls_head = nn.Linear(
            int(self._encoder.config.hidden_size), num_document_labels
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> TTorchJointOutput:
        encoded = self._encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = cast(torch.Tensor, encoded.last_hidden_state)
        token_logits = self._token_head(self._dropout(hidden))
        # Masked mean не требует специального CLS-токена у выбранного encoder.
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        document_logits = self._cls_head(self._dropout(pooled))
        return TTorchJointOutput(logits=token_logits, cls_logits=document_logits)


class PretrainedModelAdapter(BaseModel):
    """Адаптирует JointNERAndTopicModel к спискам BaseModel без обучения."""

    def __init__(self, model: JointNERAndTopicModel) -> None:
        self._model = model

    @overload
    def infer(
        self, batch_input: list[BaseBertModelInput]
    ) -> list[BaseBertModelOutput]: ...

    @overload
    def infer(
        self, batch_input: list[BaseBertModelLabeldInput]
    ) -> list[BaseBertModelOutput]: ...

    def infer(
        self,
        batch_input: list[BaseBertModelInput] | list[BaseBertModelLabeldInput],
    ) -> list[BaseBertModelOutput]:
        predictions: list[BaseBertModelOutput] = []
        was_training = self._model.training
        device = next(self._model.parameters()).device
        self._model.eval()
        try:
            with torch.no_grad():
                for document in batch_input:
                    # Адаптер BaseModel обрабатывает запись без collator и padding.
                    output: TTorchJointOutput = self._model.forward(
                        input_ids=torch.tensor(
                            [document["input_ids"]],
                            dtype=torch.long,
                            device=device,
                        ),
                        attention_mask=torch.tensor(
                            [document["attention_mask"]],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                    tags = cast(
                        list[int], output.logits.argmax(dim=-1)[0].cpu().tolist()
                    )
                    text_classes = cast(
                        list[int], (output.cls_logits[0] >= 0).long().cpu().tolist()
                    )
                    if "labels" in document:
                        labels = cast(BaseBertModelLabeldInput, document)["labels"]
                        tags = [
                            tag
                            for tag, label in zip(tags, labels)
                            if label != IGNORE_LABEL
                        ]
                    predictions.append(
                        BaseBertModelOutput(text_classes=text_classes, token_tags=tags)
                    )
        finally:
            self._model.train(was_training)
        return predictions

    def save(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), target / "model.pt")

    def load(self, path: str) -> None:
        device = next(self._model.parameters()).device
        self._model.load_state_dict(
            torch.load(Path(path) / "model.pt", map_location=device, weights_only=True)
        )
