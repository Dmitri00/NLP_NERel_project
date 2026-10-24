from pathlib import Path
from typing import cast

import torch
from torch import nn
from transformers import PreTrainedModel

from .interfaces import (
    BaseBatchModel,
    BaseBatchPrediction,
    BaseBatchTokenized,
    BaseModelType,
)


class RubertBatchModel(BaseBatchModel, nn.Module):
    """Предобученный encoder и две случайно инициализированные головы."""

    def __init__(
        self,
        model_type: BaseModelType,
        encoder: PreTrainedModel,
        num_token_tags: int,
        num_topics: int,
    ) -> None:
        nn.Module.__init__(self)
        self.model_type = model_type
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        device = next(encoder.parameters()).device
        self._token_head = nn.Linear(hidden_size, num_token_tags).to(device)
        self._topic_head = nn.Linear(hidden_size, num_topics).to(device)

    def logits(
        self, batch_input: BaseBatchTokenized
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Логиты полного батча с градиентами для encoder и обеих голов."""
        device = next(self.encoder.parameters()).device
        encoded = self.encoder(
            input_ids=torch.tensor(
                batch_input.input_ids, dtype=torch.long, device=device
            ),
            attention_mask=torch.tensor(
                batch_input.attention_mask, dtype=torch.long, device=device
            ),
            return_dict=True,
        )
        hidden = cast(torch.Tensor, encoded.last_hidden_state)
        return self._token_head(hidden), self._topic_head(hidden[:, 0, :])

    def forward(self, batch_input: BaseBatchTokenized) -> BaseBatchPrediction:
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                tag_logits, topic_logits = self.logits(batch_input)
                return BaseBatchPrediction(
                    token_tags=cast(
                        list[list[int]], tag_logits.argmax(dim=-1).cpu().tolist()
                    ),
                    text_classes=cast(
                        list[list[int]], (topic_logits >= 0).long().cpu().tolist()
                    ),
                )
        finally:
            self.train(was_training)

    def save(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "token_head": self._token_head.state_dict(),
                "topic_head": self._topic_head.state_dict(),
            },
            target / "model.pt",
        )

    def load(self, path: str) -> None:
        device = next(self.encoder.parameters()).device
        state = torch.load(
            Path(path) / "model.pt", map_location=device, weights_only=True
        )
        self.encoder.load_state_dict(state["encoder"])
        self._token_head.load_state_dict(state["token_head"])
        self._topic_head.load_state_dict(state["topic_head"])
