#!/usr/bin/env python3
"""Generate tables and radar charts from matrix_scores.csv or consolidated scores.csv.

Usage: python talktopia/generate_report.py /path/to/matrix_scores.csv
Requires matplotlib. Outputs: <project root>/reports/<CSV parent name>/.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


# Keep the original SOTOPIA order and fixed score ranges.
DIMENSIONS = {
    "believability": (0, 10),
    "relationship": (-5, 5),
    "knowledge": (0, 10),
    "secret": (-10, 0),
    "social_rules": (-10, 0),
    "financial_and_material_benefits": (-5, 5),
    "goal": (0, 10),
}
SCORES = (*DIMENSIONS, "overall_score")
MODEL_NAMES = {
    "talktopia-agent-qwen35-9b": "Qwen 3.5 9B",
    "talktopia-agent-deepseek-r1-8b": "DeepSeek R1 8B",
    "talktopia-agent-llama31-8b": "Llama 3.1 8B",
    "talktopia-agent-ministral3-8b": "Ministral 3 8B",
}


def label(model):
    return MODEL_NAMES.get(model, model)


def dimension_label(name):
    return "goal completion" if name == "goal" else name.replace("_", " ")


def read_scores(path):
    required = {"pair_id", "episode_id", "agent_index", "agent1_model", "agent2_model", *SCORES}
    rows = []
    seen = set()
    episode_pairs = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"필수 열 누락: {', '.join(sorted(missing))}")
        for line, raw in enumerate(reader, start=2):
            try:
                if any(raw.get(key) is None or not raw[key].strip() for key in required):
                    raise ValueError("필수 값이 비어 있습니다")
                role = raw["agent_index"].strip()
                if role not in ("1", "2"):
                    raise ValueError("agent_index는 1 또는 2여야 합니다")
                pair = (raw["agent1_model"].strip(), raw["agent2_model"].strip())
                episode = (raw["pair_id"].strip(), raw["episode_id"].strip())
                identity = (*episode, role)
                if identity in seen:
                    raise ValueError(f"중복 평가 행: {identity}")
                if episode in episode_pairs and episode_pairs[episode] != pair:
                    raise ValueError("같은 에피소드의 모델 조합이 서로 다릅니다")
                scores = {name: float(raw[name]) for name in SCORES}
                if not all(math.isfinite(value) for value in scores.values()):
                    raise ValueError("점수는 유한한 숫자여야 합니다")
                for name, (low, high) in DIMENSIONS.items():
                    if not low <= scores[name] <= high:
                        raise ValueError(f"{name} 점수가 범위 [{low}, {high}] 밖입니다")
                if not math.isclose(scores["overall_score"], fmean(scores[n] for n in DIMENSIONS), abs_tol=1e-6):
                    raise ValueError("overall_score가 원점수 7개의 평균과 다릅니다")
                rows.append({"pair": pair, "role": int(role), "model": pair[int(role) - 1], **scores})
                seen.add(identity)
                episode_pairs[episode] = pair
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV {line}행: {exc}") from exc
    if not rows:
        raise ValueError("CSV에 평가 행이 없습니다")
    return rows


def mean(rows, score):
    return fmean(row[score] for row in rows) if rows else None


def aggregate(rows):
    preferred_order = {model: index for index, model in enumerate(MODEL_NAMES)}
    models = sorted({model for row in rows for model in row["pair"]},
                    key=lambda model: (preferred_order.get(model, len(MODEL_NAMES)), model))
    by_model = defaultdict(list)
    by_role = defaultdict(list)
    by_pair = defaultdict(list)
    by_opponent = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
        by_role[row["model"], row["role"]].append(row)
        by_pair[row["pair"]].append(row)
        opponent = row["pair"][2 - row["role"]]
        by_opponent[row["model"], opponent].append(row)

    model_table = []
    for name in SCORES:
        bounds = DIMENSIONS.get(name)
        score_range = f"{bounds[0]} ~ {bounds[1]}" if bounds else "원점수 7개 평균"
        model_table.append([dimension_label(name), score_range, *[mean(by_model[m], name) for m in models]])

    pair_tables = []
    for fixed_role in (0, 1):
        table = []
        for fixed in models:
            for other in models:
                pair = (fixed, other) if fixed_role == 0 else (other, fixed)
                selected = by_pair[pair]
                a1 = [r for r in selected if r["role"] == 1]
                a2 = [r for r in selected if r["role"] == 2]
                table.append([label(pair[0]), label(pair[1]), mean(a1, "goal"), mean(a2, "goal"),
                              mean(selected, "overall_score"), len(a1), len(a2)])
        pair_tables.append(table)

    goal_table = [[label(m), *[mean(by_opponent[m, other], "goal") for other in models]] for m in models]
    goal_counts = [[label(m), *[len(by_opponent[m, other]) for other in models]] for m in models]
    model_counts = [[label(m), len(by_model[m]), len(by_role[m, 1]), len(by_role[m, 2])] for m in models]
    return models, by_model, by_role, model_table, pair_tables, goal_table, goal_counts, model_counts


def display(value):
    if value is None:
        return "N/A"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def table_html(headers, rows, *, numeric_columns=(), compare="columns", group_size=None):
    """Highlight extrema within each row, group of rows, or the whole matrix."""
    classes = {}
    groups = []
    if compare == "rows":
        groups = [[(r, c) for c in numeric_columns] for r in range(len(rows))]
    elif compare == "matrix":
        groups = [[(r, c) for r in range(len(rows)) for c in numeric_columns]]
    else:
        size = group_size or len(rows)
        groups = [[(r, c) for r in range(start, min(start + size, len(rows)))]
                  for start in range(0, len(rows), size) for c in numeric_columns]
    for group in groups:
        # Compare displayed values so equal-looking cells get equal emphasis.
        values = {(r, c): round(rows[r][c], 3) for r, c in group if rows[r][c] is not None}
        if not values or min(values.values()) == max(values.values()):
            continue
        low, high = min(values.values()), max(values.values())
        for position, value in values.items():
            classes[position] = "best" if value == high else "worst" if value == low else ""
    result = ["<div class='table-scroll'><table><thead><tr>"]
    result.extend(f"<th>{html.escape(str(h))}</th>" for h in headers)
    result.append("</tr></thead><tbody>")
    for r, row in enumerate(rows):
        divider = " class='group-start'" if group_size and r % group_size == 0 else ""
        result.append(f"<tr{divider}>")
        for c, value in enumerate(row):
            result.append(f"<td class='{classes.get((r, c), '')}'>{html.escape(display(value))}</td>")
        result.append("</tr>")
    result.append("</tbody></table></div>")
    return "".join(result)


def save_table(output, name, headers, rows):
    with (output / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(headers)
        writer.writerows([[display(value) for value in row] for row in rows])


def make_radars(output, models, by_model, by_role):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles = [i * 2 * math.pi / len(DIMENSIONS) for i in range(len(DIMENSIONS))]
    closed_angles = angles + angles[:1]
    colors = plt.get_cmap("tab10" if len(models) <= 10 else "tab20")
    tick_labels = [dimension_label(n).replace("financial and material benefits", "financial / material\nbenefits") for n in DIMENSIONS]
    charts = []
    for suffix, title, role in (("all", "All roles", None), ("a1", "Agent 1", 1), ("a2", "Agent 2", 2)):
        fig, ax = plt.subplots(figsize=(11, 8), subplot_kw={"projection": "polar"})
        fig.subplots_adjust(left=0.14, right=0.72, top=0.85, bottom=0.14)
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles, tick_labels, fontsize=10)
        ax.tick_params(axis="x", pad=15)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontsize=8)
        for index, model in enumerate(models):
            selected = by_model[model] if role is None else by_role[model, role]
            if not selected:
                continue
            values = [100 * (mean(selected, name) - low) / (high - low) for name, (low, high) in DIMENSIONS.items()]
            color = colors(index % colors.N)
            ax.plot(closed_angles, values + values[:1], color=color, linewidth=1.8, marker="o", markersize=3,
                    label=f"{label(model)} (n={len(selected)})")
            ax.fill(closed_angles, values + values[:1], color=color, alpha=0.04)
        ax.set_title(f"SOTOPIA evaluation — {title}", pad=38)
        if ax.lines:
            ax.legend(loc="upper left", bbox_to_anchor=(1.22, 1.12), frameon=False)
        fig.text(0.5, 0.035, "Fixed-range normalization: 100 × (score − minimum) / (maximum − minimum). Higher is better.",
                 ha="center", fontsize=9)
        name = f"radar_{suffix}"
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        charts.append((name, title))
    return charts


def generate_report(source, output, rows):
    models, by_model, by_role, model_table, pair_tables, goal_table, goal_counts, model_counts = aggregate(rows)
    output.mkdir(parents=True, exist_ok=True)
    charts = make_radars(output, models, by_model, by_role)
    model_headers = [label(m) for m in models]
    pair_headers = ["Agent 1", "Agent 2", "A1 goal", "A2 goal", "전체 평균", "A1 표본 수", "A2 표본 수"]
    sections = [
        ("model_scores", "모델별 SOTOPIA-EVAL 평균 점수", ["평가 축", "범위", *model_headers], model_table,
         dict(numeric_columns=range(2, len(models) + 2), compare="rows")),
        ("pairs_by_a1", "조합별 점수 — A1 기준", pair_headers, pair_tables[0],
         dict(numeric_columns=(2, 3, 4), group_size=len(models))),
        ("pairs_by_a2", "조합별 점수 — A2 기준", pair_headers, pair_tables[1],
         dict(numeric_columns=(2, 3, 4), group_size=len(models))),
        ("goal_by_opponent", "상대 모델별 목표 점수 — 양쪽 역할 통합", ["평가 대상 ↓ / 상대 →", *model_headers], goal_table,
         dict(numeric_columns=range(1, len(models) + 1), compare="matrix")),
        ("model_counts", "모델별 표본 수", ["모델", "전체", "A1", "A2"], model_counts, {}),
        ("goal_counts", "상대 모델별 목표 점수 표본 수", ["평가 대상 ↓ / 상대 →", *model_headers], goal_counts, {}),
    ]
    parts = ["""<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOTOPIA 평가 보고서</title><style>
body{font-family:system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#202630;line-height:1.6}
h2{margin-top:44px}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #dce1e8;padding:9px 12px;text-align:right;white-space:nowrap}
th{background:#f1f4f8}td:first-child,th:first-child{text-align:left}.group-start td{border-top:2px solid #aab4c2}
.best{color:#1463ca;font-weight:700}.worst{color:#c42d34;font-weight:700}img{max-width:100%;height:auto}
p,li{overflow-wrap:anywhere}a{color:#1463ca}code{background:#f1f4f8;padding:2px 5px}
</style><h1>SOTOPIA 평가 보고서</h1>""",
        f"<p>데이터: <strong>{html.escape(source.parent.name)}</strong><br>"
        f"입력: <code>{html.escape(str(source))}</code><br>평가 행: {len(rows):,}개</p>",
        """<p>평균은 CSV의 개별 평가 행을 동일한 가중치로 계산합니다. 역할·조합별 표본 수가 다르면
표본이 많은 쪽의 영향이 커집니다. overall score와 조합별 전체 평균은 정규화하지 않은 원점수를 사용합니다.
전체 평균은 해당 조합의 모든 평가 행에 있는 overall_score의 평균입니다.</p>
<p>상대 모델별 목표 점수는 대상 모델의 A1·A2 역할을 합쳐 계산합니다.
같은 모델끼리 대화한 경우 두 에이전트의 점수를 각각 한 표본으로 포함합니다.
없는 조합·역할은 N/A로 표시하고 평균에서 제외합니다. CSV에 없는 실패·미완료 평가의 수는 알 수 없습니다.</p>
<p>소수 셋째 자리까지 표시합니다. 파랑은 최고, 빨강은 최저이며 동점은 함께 강조합니다.
모델별 표는 각 행, 조합별 표는 각 묶음의 점수 열, 목표 점수 행렬은 전체 셀을 비교합니다.
비교 값이 모두 같으면 강조하지 않습니다.</p>""",
    ]
    parts.append("<h2>모델 이름</h2><ul>" + "".join(
        f"<li>{html.escape(label(m))}: <code>{html.escape(m)}</code></li>" for m in models) + "</ul>")
    for name, title, headers, table, options in sections:
        save_table(output, name, headers, table)
        parts.extend([f"<h2>{title}</h2><p><a href='{name}.csv'>CSV 다운로드</a></p>", table_html(headers, table, **options)])
    parts.append("<h2>Radar Chart</h2><p>7개 축을 고정 범위 기준으로 0~100으로 변환합니다: "
                 "100 × (점수 − 최솟값) / (최댓값 − 최솟값). 원래 범위는 모델별 평균 표에 표시했습니다. "
                 "바깥쪽일수록 높은 점수입니다. overall_score는 차트에서 제외하며, 표본이 없는 모델·역할은 그리지 않습니다.</p>")
    for name, title in charts:
        parts.append(f"<h3>{title}</h3><p><a href='{name}.png'>PNG</a> · <a href='{name}.svg'>SVG</a></p>"
                     f"<img src='{name}.svg' alt='{title}: 모델별 7개 평가 축 Radar Chart'>")
    parts.append("</html>")
    (output / "report.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="matrix_scores.csv 또는 통합 scores.csv 경로")
    args = parser.parse_args()
    source = args.csv_path.expanduser().resolve()
    output = Path(__file__).resolve().parents[1] / "reports" / (source.parent.name or "matrix_report")
    try:
        rows = read_scores(source)
        generate_report(source, output, rows)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"오류: {exc}\n")
    print(f"보고서: {output / 'report.html'}")
    print(f"표 CSV 6개, Radar Chart PNG 3개 · SVG 3개 생성 완료")


if __name__ == "__main__":
    main()
