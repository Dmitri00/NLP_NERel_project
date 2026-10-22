from html import escape
from pathlib import Path

import polars as pl


def generate_label_report(
    label_metrics: pl.DataFrame,
    label_facts: pl.DataFrame,
    label_total_documents: int,
) -> tuple[str, str, Path]:
    label_metric_rows = label_metrics.to_dicts()
    label_bar_rows = "".join(
        f'<div class="cls-bar-row"><span>{rank}. {escape(row["label"])}</span>'
        f'<div class="cls-track"><div style="width:{row["documents_share"]:.6f}%"></div></div>'
        f"<span>{row['documents_share']:.2f}% · {row['documents']:,}</span></div>"
        for rank, row in enumerate(label_metric_rows, start=1)
    )
    label_table_rows = "".join(
        f'<tr><th scope="row">{escape(row["label"])}</th>'
        f"<td>{row['documents']:,}</td><td>{row['documents_share']:.2f}%</td></tr>"
        for row in label_metric_rows
    )
    label_fact_rows = "".join(
        f"<tr><td>{row['id']}</td><td>{escape(row['label'])}</td><td>{row['value']}</td></tr>"
        for row in label_facts.head(12).to_dicts()
    )

    label_report_fragment = f"""
<style>
.cls-report {{font:14px/1.5 system-ui,sans-serif;color:#172033;background:#f5f7fb;padding:24px;border-radius:12px;max-width:1200px;margin:auto}}
.cls-report * {{box-sizing:border-box}}
.cls-report h1,.cls-report h2 {{margin:0 0 16px}}
.cls-report h2 {{font-size:19px}}
.cls-cards {{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}}
.cls-card,.cls-panel {{background:white;border:1px solid #dce3ef;border-radius:10px;padding:18px}}
.cls-card strong {{display:block;font-size:27px;color:#2256b8}}
.cls-columns {{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:16px}}
.cls-bar-row {{display:grid;grid-template-columns:165px minmax(24px,1fr) 100px;align-items:center;gap:8px;font-size:11px;margin:8px 0}}
.cls-bar-row>span:last-child {{text-align:right;font-variant-numeric:tabular-nums}}
.cls-track {{height:12px;background:#edf1f8;border-radius:3px;overflow:hidden}}
.cls-track>div {{height:100%;background:#3873da}}
.cls-scroll {{overflow:auto}}
.cls-report table {{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
.cls-report th,.cls-report td {{padding:8px 10px;border-bottom:1px solid #e5eaf2;text-align:right;white-space:nowrap}}
.cls-report th:first-child,.cls-report td:first-child {{text-align:left}}
.cls-report thead th {{background:#eaf0fa}}
.cls-report tbody tr:nth-child(even) {{background:#f7f9fc}}
.cls-note {{color:#475569}}
.cls-report details {{margin-top:20px}}
@media (max-width:480px) {{.cls-report {{padding:12px}} .cls-panel {{padding:10px}} .cls-bar-row {{grid-template-columns:150px minmax(16px,1fr) 85px;gap:4px;font-size:10px}}}}
</style>
<article class="cls-report" lang="ru">
<h1>NEREL — характеристики документов (CLS)</h1>
<p class="cls-note">Многометочная классификация: у документа может быть несколько характеристик.
Одна строка фактов — пара id–label со значением value = 1.</p>
<div class="cls-cards">
<div class="cls-card">Всего документов<strong>{label_total_documents:,}</strong></div>
<div class="cls-card">Заданных характеристик<strong>{label_metrics.height}</strong></div>
<div class="cls-card">Активных пар id–label<strong>{label_facts.height:,}</strong></div>
</div>
<p class="cls-note">Для каждой характеристики: доля = 100 × число уникальных документов с ней / {label_total_documents:,}.
Сумма долей может превышать 100%. Учитываются все 30 характеристик, включая отсутствующие.</p>
<div class="cls-columns">
<section class="cls-panel"><h2>Доля документов, % · число документов</h2>
<p class="cls-note">Единая шкала 0–100%; порядок по убыванию частоты.</p>{label_bar_rows}</section>
<section class="cls-panel"><h2>Метрики по всем характеристикам</h2>
<div class="cls-scroll"><table><thead><tr><th>Характеристика</th><th>Документов</th><th>Доля</th></tr></thead>
<tbody>{label_table_rows}</tbody></table></div></section>
</div>
<details><summary>Таблица фактов: первые 12 строк из {label_facts.height:,}</summary>
<p>Полная таблица доступна в label_facts; метрики — в label_metrics.</p>
<div class="cls-scroll"><table><thead><tr><th>id</th><th>label</th><th>value</th></tr></thead>
<tbody>{label_fact_rows}</tbody></table></div></details>
</article>
"""
    label_report_html = (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>NEREL — EDA характеристик документов</title></head><body>"
        + label_report_fragment
        + "</body></html>"
    )
    label_report_path = Path("reports/nerel_task2.html")
    label_report_path.parent.mkdir(parents=True, exist_ok=True)
    label_report_path.write_text(label_report_html, encoding="utf-8")
    return label_report_fragment, label_report_html, label_report_path
