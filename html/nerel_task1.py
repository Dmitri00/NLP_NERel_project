from html import escape
from pathlib import Path

import polars as pl


def generate_tag_report(
    tag_metrics: pl.DataFrame,
    tag_facts: pl.DataFrame,
    total_documents: int,
    total_tokens: int,
) -> tuple[str, str, Path]:
    metric_rows = tag_metrics.to_dicts()

    def metric_bars(
        column: str, count_column: str, *, show_cumulative: bool = False
    ) -> str:
        # Каждую диаграмму сортируем по её частоте; нумерация начинается с 1.
        rows = tag_metrics.sort(
            [count_column, "tag"], descending=[True, False]
        ).to_dicts()
        bars = []
        for rank, row in enumerate(rows, start=1):
            cumulative_bar = ""
            cumulative_value = ""
            if show_cumulative:
                cumulative_bar = (
                    '<div class="nerel-track nerel-cumulative-track">'
                    f'<div style="width:{row["cumulative_share"]:.6f}%"></div></div>'
                )
                cumulative_value = (
                    '<br><span class="nerel-cumulative-value">'
                    f"Σ {row['cumulative_share']:.3f}%</span>"
                )
            bars.append(
                f'<div class="nerel-bar-row"><span>{rank}. {escape(row["tag"])}</span>'
                '<div class="nerel-tracks"><div class="nerel-track">'
                f'<div style="width:{row[column]:.6f}%"></div></div>{cumulative_bar}</div>'
                f"<span>{row[column]:.3f}% · {row[count_column]:,}{cumulative_value}</span></div>"
            )
        return "".join(bars)

    metric_table_rows = "".join(
        f'<tr><th scope="row">{escape(row["tag"])}</th>'
        f"<td>{row['documents']:,}</td><td>{row['documents_share']:.3f}%</td>"
        f"<td>{row['tokens']:,}</td><td>{row['tokens_share']:.3f}%</td>"
        f"<td>{row['cumulative_share']:.3f}%</td></tr>"
        for row in metric_rows
    )
    fact_table_rows = "".join(
        f"<tr><td>{escape(row['tag'])}</td><td>{row['id']}</td><td>{row['count']}</td></tr>"
        for row in tag_facts.head(12).to_dicts()
    )

    report_fragment = f"""
<style>
.nerel-report {{font:14px/1.5 system-ui,sans-serif;color:#172033;background:#f5f7fb;padding:24px;border-radius:12px;max-width:1400px;margin:auto}}
.nerel-report * {{box-sizing:border-box}}
.nerel-report h1,.nerel-report h2 {{color:#172033;margin:0 0 16px}}
.nerel-report h2 {{font-size:19px}}
.nerel-report .nerel-cards {{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}}
.nerel-card,.nerel-panel {{background:white;border:1px solid #dce3ef;border-radius:10px;padding:18px}}
.nerel-card strong {{display:block;font-size:27px;color:#2256b8}}
.nerel-charts {{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;margin-bottom:20px}}
.nerel-bar-row {{display:grid;grid-template-columns:185px minmax(40px,1fr) 145px;align-items:center;gap:8px;font-size:11px;margin:5px 0}}
.nerel-bar-row>span:last-child {{text-align:right;font-variant-numeric:tabular-nums}}
.nerel-track {{height:12px;background:#edf1f8;border-radius:3px;overflow:hidden}}
.nerel-track>div {{height:100%;background:#3873da;min-width:1px}}
.nerel-track+.nerel-track {{margin-top:4px}}
.nerel-cumulative-track>div {{background:#d97706}}
.nerel-cumulative-value {{color:#a65308}}
.nerel-scroll {{overflow:auto;max-height:650px}}
.nerel-report table {{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}}
.nerel-report th,.nerel-report td {{padding:8px 12px;border-bottom:1px solid #e5eaf2;text-align:right;white-space:nowrap}}
.nerel-report th:first-child,.nerel-report td:first-child {{text-align:left}}
.nerel-report thead th {{position:sticky;top:0;background:#eaf0fa;color:#172033}}
.nerel-report tbody tr:nth-child(even) {{background:#f7f9fc}}
.nerel-note {{color:#475569}}
.nerel-report details {{margin-top:20px}}
</style>
<article class="nerel-report" lang="ru">
<h1>NEREL — частоты BIO-меток</h1>
<p class="nerel-note">Исходные слова до субтокенизации. B- и I- учитываются отдельно; O исключена из частот и общего числа токенов.</p>
<div class="nerel-cards">
<div class="nerel-card">Документов<strong>{total_documents:,}</strong></div>
<div class="nerel-card">Исходных токенов без O<strong>{total_tokens:,}</strong></div>
<div class="nerel-card">BIO-меток без O<strong>{tag_metrics.height}</strong></div>
<div class="nerel-card">Пар документ–метка<strong>{tag_facts.height:,}</strong></div>
</div>
<div class="nerel-charts">
<section class="nerel-panel"><h2>Доля токенов, % · число токенов</h2>
<p class="nerel-note">Синий — доля метки, оранжевый (Σ) — накопленная доля. Обе шкалы — 0–100%. Метки упорядочены по убыванию числа токенов.</p>
{metric_bars("tokens_share", "tokens", show_cumulative=True)}</section>
<section class="nerel-panel"><h2>Документы с меткой, % · число документов</h2>
<p class="nerel-note">Метки упорядочены по убыванию числа документов с меткой.</p>
{metric_bars("documents_share", "documents")}</section>
</div>
<section class="nerel-panel"><h2>Метрики по всем меткам</h2>
<p class="nerel-note">Доли документов могут в сумме превышать 100%: один документ содержит несколько меток.
Сумма долей токенов: {tag_metrics["tokens_share"].sum():.2f}%. Порядок — по убыванию числа токенов. Накопленная доля — сумма долей токенов от первой строки до текущей.</p>
<div class="nerel-scroll"><table><thead><tr><th>Метка</th><th>Документов</th><th>Доля документов</th><th>Токенов</th><th>Доля токенов</th><th>Накопленная доля токенов</th></tr></thead>
<tbody>{metric_table_rows}</tbody></table></div></section>
<details><summary>Таблица фактов: первые 12 строк из {tag_facts.height:,}</summary>
<p>Одна строка — метка в документе, count &gt; 0. Полная таблица доступна в tag_facts.</p>
<table><thead><tr><th>tag</th><th>id</th><th>count</th></tr></thead><tbody>{fact_table_rows}</tbody></table></details>
</article>
"""
    report_html = (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>NEREL — EDA BIO-меток</title></head><body>"
        + report_fragment
        + "</body></html>"
    )
    report_path = Path("reports/nerel_task1.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_html, encoding="utf-8")
    return report_fragment, report_html, report_path
