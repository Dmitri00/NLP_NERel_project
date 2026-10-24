from base import BaseDocument, BaseModel, BaseTextPrediction, BaseTextTagFabric

from .interfaces import (
    BaseBatchModel,
    BaseBatchPrediction,
    BaseBatchTokenized,
    BaseBatchTokenizer,
)


class BaseSubwordToWordAligner:
    """Выравнивает одно окно на документ, дополняя непокрытые слова тегами O."""

    def __init__(self, tag_fabric: BaseTextTagFabric) -> None:
        self._tag_fabric = tag_fabric

    def align(
        self,
        batch_tokenized: BaseBatchTokenized,
        batch_prediction: BaseBatchPrediction,
        input: list[BaseDocument],
    ) -> list[BaseTextPrediction]:
        predictions: list[BaseTextPrediction] = []
        for batch_index, document in enumerate(input):
            first_tags: dict[int, int] = {}
            tags = batch_prediction["token_tags"][batch_index]
            word_ids = batch_tokenized.get_word_ids(batch_index)
            mask = batch_tokenized.attention_mask[batch_index]
            for word_id, tag_id, active in zip(word_ids, tags, mask):
                if word_id is not None and active:
                    # Сохраняем тег первого субтокена слова.
                    first_tags.setdefault(word_id, tag_id)
            predictions.append(
                BaseTextPrediction(
                    input_text=BaseDocument(text=document.text.copy()),
                    tag_prediction=[
                        self._tag_fabric.by_id(first_tags[index])
                        if index in first_tags
                        else self._tag_fabric.by_value("O")
                        for index in range(len(document.text))
                    ],
                    topic_prediction=batch_prediction["text_classes"][
                        batch_index
                    ].copy(),
                )
            )
        return predictions


class BatchModelAdapter(BaseModel):
    def __init__(
        self,
        model: BaseBatchModel,
        tokenizer: BaseBatchTokenizer,
        aligner: BaseSubwordToWordAligner,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._aligner = aligner

    def infer(self, documents: list[BaseDocument]) -> list[BaseTextPrediction]:
        predictions: list[BaseTextPrediction] = []
        for document in documents:
            # Документы обрабатываются отдельно: word_ids локальны.
            tokenized = self._tokenizer.tokenize([document])
            prediction = self._model.forward(tokenized)
            predictions.extend(self._aligner.align(tokenized, prediction, [document]))
        return predictions

    def infer_batch(
        self, batch: BaseBatchTokenized, documents: list[BaseDocument]
    ) -> list[BaseTextPrediction]:
        """Инференс уже токенизированного батча с выравниванием к словам."""
        prediction = self._model.forward(batch)
        return self._aligner.align(batch, prediction, documents)

    def save(self, path: str) -> None:
        self._model.save(path)

    def load(self, path: str) -> None:
        self._model.load(path)
