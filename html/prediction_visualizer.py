from html import escape
from zlib import crc32

from IPython.display import HTML, display

from base import (
    BaseDocument,
    BaseMetricEvaluator,
    BaseNerelExample,
    BaseTextPrediction,
    BaseVisualizer,
)


def _tag_color(tag: str) -> str:
    if tag == "O":
        return "#f1f5f9"
    entity = tag.removeprefix("B-").removeprefix("I-")
    hue = crc32(entity.encode("utf-8")) % 360
    return f"hsl({hue} 65% 85%)"


class HtmlPredictionVisualizer(BaseVisualizer):
    def __init__(
        self,
        topic_names: list[str],
        title: str,
        token_evaluator: BaseMetricEvaluator,
        topic_evaluator: BaseMetricEvaluator,
    ) -> None:
        self._topic_names = topic_names.copy()
        self._title = title
        self._token_evaluator = token_evaluator
        self._topic_evaluator = topic_evaluator

    def _metric_summary(self, values: dict[str, float] | None, target: str) -> str:
        if values is None:
            return '<p class="np-note">Метрики недоступны без истинной разметки.</p>'
        return (
            '<div class="np-metrics">'
            + "".join(
                f"<span>{label}: <strong>{values[f'{key}-{target}']:.4f}</strong></span>"
                for label, key in (
                    ("Precision", "precision"),
                    ("Recall", "recall"),
                    ("F1", "f1-macro"),
                )
            )
            + "</div>"
        )

    def visualize(
        self,
        input_text: BaseNerelExample | BaseDocument,
        prediction: BaseTextPrediction,
    ) -> None:
        if isinstance(input_text, BaseDocument):
            tokens = input_text.text
            actual_tags = None
            actual_topics = None
            token_metrics = None
            topic_metrics = None
        else:
            tokens = input_text["tokens"].text
            actual_tags = input_text["tags"]
            actual_topics = input_text["cls_vec"]
            token_metrics = self._token_evaluator.eval([input_text], [prediction])
            topic_metrics = self._topic_evaluator.eval([input_text], [prediction])

        topic_rows: list[str] = []
        for index, name in enumerate(self._topic_names):
            actual = actual_topics[index] if actual_topics is not None else None
            predicted = prediction["topic_prediction"][index]
            if actual == 0 and predicted == 0:
                continue
            status = (
                "Нет эталона"
                if actual is None
                else "Совпадает"
                if actual == predicted
                else "Лишняя"
                if predicted
                else "Пропущена"
            )
            row_class = (
                "np-mismatch" if actual is not None and actual != predicted else ""
            )
            actual_label = "—" if actual is None else "Да" if actual else "Нет"
            predicted_label = "Да" if predicted else "Нет"
            topic_rows.append(
                f'<tr class="{row_class}"><th scope="row">{escape(name)}</th>'
                f"<td>{actual_label}</td><td>{predicted_label}</td><td>{status}</td></tr>"
            )

        words: list[str] = []
        for index, token in enumerate(tokens):
            actual_tag = actual_tags[index].value if actual_tags is not None else None
            predicted_tag = prediction["tag_prediction"][index].value
            actual_label = actual_tag if actual_tag is not None else "—"
            actual_color = (
                _tag_color(actual_tag) if actual_tag is not None else "#ffffff"
            )
            predicted_color = _tag_color(predicted_tag)
            word_class = (
                "np-word np-different"
                if actual_tag is not None and actual_tag != predicted_tag
                else "np-word"
            )
            tooltip = escape(
                f"Истина: {actual_label}; прогноз: {predicted_tag}", quote=True
            )
            words.append(
                f'<span class="{word_class}" title="{tooltip}">'
                f'<span class="np-tag" style="background:{actual_color}">'
                f"{escape(actual_label)}</span>"
                '<span class="np-token" style="background:linear-gradient('
                f'to bottom,{actual_color} 50%,{predicted_color} 50%)">'
                f"{escape(token)}</span>"
                f'<span class="np-tag" style="background:{predicted_color}">'
                f"{escape(predicted_tag)}</span></span>"
            )

        entity_types = sorted(
            {
                tag.value.removeprefix("B-").removeprefix("I-")
                for tag in (actual_tags or []) + prediction["tag_prediction"]
                if tag.value != "O"
            }
        )
        legend = " ".join(
            f'<span class="np-legend-item" style="background:{_tag_color(entity)}">'
            f"{escape(entity)}</span>"
            for entity in entity_types
        )
        display(
            HTML(
                """
<style>
.nerel-prediction {font:14px/1.5 system-ui,sans-serif;color:#172033;
    background:#f5f7fb;padding:22px;border-radius:12px;max-width:1200px;margin:16px auto}
.nerel-prediction * {box-sizing:border-box}
.nerel-prediction h1 {font-size:23px;margin:0 0 18px;color:#172033}
.nerel-prediction h2 {font-size:18px;margin:0 0 12px;color:#172033}
.nerel-prediction section {background:white;border:1px solid #dce3ef;
    border-radius:10px;padding:18px;margin-top:16px}
.nerel-prediction .np-scroll {overflow:auto;max-height:360px}
.nerel-prediction table {border-collapse:collapse;width:100%;font-size:13px}
.nerel-prediction th,.nerel-prediction td {padding:7px 12px;text-align:left;
    border-bottom:1px solid #e2e8f0;white-space:nowrap;color:#172033}
.nerel-prediction thead th {position:sticky;top:0;background:#eaf0fa}
.nerel-prediction .np-mismatch {background:#fff3e4}
.nerel-prediction .np-note {color:#475569;margin:8px 0 16px}
.nerel-prediction .np-metrics {display:flex;flex-wrap:wrap;gap:8px 24px;
    padding:10px 12px;margin:12px 0;background:#eaf0fa;border-radius:6px;
    font-variant-numeric:tabular-nums}
.nerel-prediction .np-text {line-height:1.5;overflow-wrap:anywhere}
.nerel-prediction .np-word {display:inline-grid;grid-template-rows:auto auto auto;
    vertical-align:middle;text-align:center;margin:5px 2px;border-radius:4px;
    max-width:100%;overflow-wrap:anywhere}
.nerel-prediction .np-tag {font:9px/1.5 ui-monospace,monospace;padding:1px 4px}
.nerel-prediction .np-token {font-size:15px;padding:3px 4px}
.nerel-prediction .np-different {outline:1px dashed #9a3412;outline-offset:1px}
.nerel-prediction .np-legend-item {display:inline-block;padding:2px 7px;
    border-radius:4px;margin:3px;font-size:11px}
@media(max-width:600px) {.nerel-prediction {padding:10px}
    .nerel-prediction section {padding:10px}}
</style>
"""
                '<article class="nerel-prediction" lang="ru">'
                f"<h1>{escape(self._title)}</h1>"
                '<section aria-label="Сравнение характеристик">'
                "<h2>1. Характеристики документа</h2>"
                + self._metric_summary(topic_metrics, "topics")
                + '<p class="np-note">Метрики текущего документа: macro по всем '
                "30 характеристикам, zero_division=0. В таблице скрыты характеристики, "
                "отсутствующие и в истине, и в прогнозе.</p>"
                '<p class="np-note">Да = 1, Нет = 0. Оранжевый фон — расхождение. '
                "Лишняя: только в прогнозе; пропущена: только в истине. "
                "При вводе BaseDocument истинная разметка недоступна (—).</p>"
                '<div class="np-scroll"><table><thead><tr><th>Характеристика</th>'
                "<th>Истина</th><th>Прогноз</th><th>Сравнение</th></tr></thead><tbody>"
                + (
                    "".join(topic_rows)
                    or '<tr><td colspan="4">Нет активных характеристик.</td></tr>'
                )
                + "</tbody></table></div></section>"
                '<section aria-label="Текст с разметкой">'
                "<h2>2. Текст и BIO-теги</h2>"
                + self._metric_summary(token_metrics, "tokens")
                + '<p class="np-note">Метрики текущего текста: macro по всему '
                "справочнику BIO-тегов без O, zero_division=0. "
                "Перед расчётом удалены только пары (O, O). Пары (O, сущность) "
                "и (сущность, O) учитываются как ошибки. "
                "Если после фильтрации пар нет, метрики равны 0.</p>"
                '<p class="np-note">Над словом и в верхней половине подсветки — истина; '
                "под словом и в нижней половине — прогноз. B- — начало сущности, "
                "I- — продолжение; O — вне сущности (серый). "
                "Пунктир — несовпадение тегов. Текст собран из исходных слов через пробел.</p>"
                f'<div aria-label="Типы сущностей">{legend}</div>'
                '<div class="np-text">' + " ".join(words) + "</div></section></article>"
            )
        )
