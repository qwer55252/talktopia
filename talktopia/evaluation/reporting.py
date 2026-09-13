"""Readable evaluation results and run/model-matrix score summaries."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from talktopia.utils import write_json, write_text_atomic


def write_episode_report(path, evaluated, names, responses, evaluator_model) -> None:
    from sotopia.database import SotopiaDimensions

    lines = [
        "# SOTOPIA episode evaluation",
        "",
        f"Evaluator: {evaluator_model}",
        "",
        f"| Dimension | {names[0]} (agent1) | {names[1]} (agent2) |",
        "|---|---:|---:|",
    ]
    for dimension in (*SotopiaDimensions.model_fields, "overall_score"):
        lines.append(
            f"| {dimension} | {evaluated.rewards[0][1][dimension]:g} | {evaluated.rewards[1][1][dimension]:g} |"
        )
    lines.extend(
        [
            "",
            "Overall score is the unweighted mean of the seven original scores; it is not normalized.",
            "",
        ]
    )
    for index, name in enumerate(names, start=1):
        lines.extend([f"## {name} (agent{index})", ""])
        for agent, ((dimension, score), reason) in responses:
            if agent == f"agent_{index}":
                lines.extend([f"### {dimension}: {score}", "", reason.strip(), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_evaluation_report(run_dir: Path) -> None:
    from sotopia.database import SotopiaDimensions

    progress = json.loads(
        (run_dir / "04_sotopia_eval_reevaluate_existing.json").read_text()
    )
    fields = [*SotopiaDimensions.model_fields, "overall_score"]
    rows = []
    no_speech = 0
    for result in progress["episodes"]:
        if result["status"] != "completed":
            continue
        episode = json.loads((run_dir / result["original"]).read_text())
        no_speech += result.get("source_conversation_audio") is None
        for index, (pk, reward) in enumerate(
            zip(episode["agents"], episode["rewards"]), start=1
        ):
            overall, scores = reward
            if (
                set(scores) != set(fields)
                or abs(
                    overall
                    - sum(scores[name] for name in SotopiaDimensions.model_fields) / 7
                )
                > 1e-8
            ):
                raise ValueError("Invalid scores in completed evaluation")
            rows.append(
                {
                    "episode_id": result["episode_id"],
                    "combo_id": result.get("combo_id"),
                    "env_id": result["env_id"],
                    "agent_index": index,
                    "agent_pk": pk,
                    "model": episode["models"][index],
                    **scores,
                }
            )
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "episode_id",
            "combo_id",
            "env_id",
            "agent_index",
            "agent_pk",
            "model",
            *fields,
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(run_dir / "evaluation/scores.csv", output.getvalue())
    models = {}
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        models[model] = {
            "agent_count": len(selected),
            "mean_scores": {
                name: sum(row[name] for row in selected) / len(selected)
                for name in fields
            },
        }
    size = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    report = {
        key: progress[key]
        for key in (
            "run_id",
            "status",
            "total",
            "completed",
            "failed",
            "pending",
            "elapsed_seconds",
        )
    }
    report.update(
        agent_evaluations=len(rows),
        dimension_scores=len(rows) * 7,
        no_speech_episodes=no_speech,
        models=models,
        output_bytes=size,
    )
    for key in (
        "source_total",
        "source_selected",
        "source_unselected",
        "source_completed",
        "source_failed",
        "source_excluded",
        "excluded_episodes",
        "source_no_speech_episodes",
        "evaluation_total",
    ):
        if key in progress:
            report[key] = progress[key]
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        simulation_dir = Path(json.loads(config_path.read_text())["simulation_dir"])
        simulation = json.loads((simulation_dir / "03_simulation.json").read_text())
        report.update(
            simulation_dir=str(simulation_dir),
            simulation_elapsed_seconds=simulation.get("elapsed_seconds"),
            simulation_output_bytes=sum(
                path.stat().st_size
                for path in simulation_dir.rglob("*")
                if path.is_file()
            ),
        )
    write_json(run_dir / "05_summary.json", report)
    lines = [
        "# Simulation and evaluation summary",
        "",
        f"Completed: {report['completed']}/{report['total']}; failed: {report['failed']}; pending: {report['pending']}.",
        "",
        f"Agent evaluations: {len(rows)}; dimension scores: {len(rows) * 7}; episodes without speech: {no_speech}.",
        "",
        "| Model | Agent count | " + " | ".join(fields) + " |",
        "|---|---:|" + "---:|" * len(fields),
    ]
    if "source_total" in report:
        lines[2:2] = [
            f"Source simulations: {report['source_total']}; completed: {report['source_completed']}; failed: {report['source_failed']}; excluded: {report['source_excluded']}.",
            f"Evaluation targets: {report['evaluation_total']}, including {report['source_no_speech_episodes']} episodes without speech. Source exclusions are not zero scores.",
            "",
        ]
    for model, values in models.items():
        lines.append(
            f"| {model} | {values['agent_count']} | "
            + " | ".join(f"{values['mean_scores'][name]:.4f}" for name in fields)
            + " |"
        )
    lines.extend(
        [
            "",
            "Overall scores are unweighted means of the seven original scores, not scores normalized to 10.",
            "",
            f"Evaluation attempt time: {report['elapsed_seconds']:.1f} seconds. Evaluation output size: {size} bytes.",
            "",
        ]
    )
    if "simulation_dir" in report:
        lines.extend(
            [
                f"Simulation: {report['simulation_dir']}",
                f"Simulation attempt time: {report['simulation_elapsed_seconds']} seconds. Simulation output size: {report['simulation_output_bytes']} bytes.",
                "",
            ]
        )
    if report.get("excluded_episodes"):
        lines.extend(["## Excluded episodes", ""])
        for row in report["excluded_episodes"]:
            lines.append(
                f"- {row['episode_id']}: {row['reason']} — {row.get('error') or 'No error detail recorded'}"
            )
        lines.append("")
    write_text_atomic(run_dir / "05_summary.md", "\n".join(lines))


def write_matrix_report(run_dir: Path, state: dict, pairs: list[dict]) -> None:
    """Aggregate existing pair reports without inventing scores for missing results."""
    score_rows = []
    for row in pairs:
        row.update(evaluation_completed=0, evaluation_failed=0, evaluation_excluded=0)
        pair = row
        if pair.get("run_dir"):
            path = Path(pair["run_dir"])
            sim_path = path / "03_simulation.json"
            if sim_path.exists():
                sim = json.loads(sim_path.read_text())
                if sim.get("evaluation_run"):
                    evaluation_dir = Path(sim["evaluation_run"])
                    eval_path = (
                        evaluation_dir / "04_sotopia_eval_reevaluate_existing.json"
                    )
                    if eval_path.exists():
                        evaluation = json.loads(eval_path.read_text())
                        row.update(
                            evaluation_completed=evaluation["completed"],
                            evaluation_failed=evaluation["failed"],
                            evaluation_excluded=evaluation.get("source_excluded", 0),
                            evaluation_wall_seconds=evaluation.get("wall_seconds", 0),
                        )
                    report_path = evaluation_dir / "05_summary.json"
                    if report_path.exists():
                        row["evaluation_report"] = json.loads(report_path.read_text())
                    scores_path = evaluation_dir / "evaluation/scores.csv"
                    if scores_path.exists():
                        with scores_path.open() as source:
                            pair_scores = list(csv.DictReader(source))
                        score_rows.extend(
                            {
                                "pair_id": pair["pair_id"],
                                "agent1_model": pair["agent1_model"],
                                "agent2_model": pair["agent2_model"],
                                **score,
                            }
                            for score in pair_scores
                        )
                        from sotopia.database import SotopiaDimensions

                        row["agent_scores"] = {}
                        for index in (1, 2):
                            selected = [
                                score
                                for score in pair_scores
                                if int(score["agent_index"]) == index
                            ]
                            row["agent_scores"][str(index)] = dict(
                                count=len(selected),
                                means=(
                                    {
                                        name: sum(
                                            float(score[name]) for score in selected
                                        )
                                        / len(selected)
                                        for name in (
                                            *SotopiaDimensions.model_fields,
                                            "overall_score",
                                        )
                                    }
                                    if selected
                                    else {}
                                ),
                            )
    totals = {
        key: sum(row[key] for row in pairs)
        for key in (
            "simulation_total",
            "simulation_completed",
            "simulation_failed",
            "evaluation_completed",
            "evaluation_failed",
            "evaluation_excluded",
        )
    }
    totals["simulation_pending"] = (
        totals["simulation_total"]
        - totals["simulation_completed"]
        - totals["simulation_failed"]
    )
    report = dict(
        status=state["status"],
        pairs=pairs,
        wall_seconds=state.get("wall_seconds", 0),
        **totals,
    )
    write_json(run_dir / "matrix_summary.json", report)
    if score_rows:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
        write_text_atomic(run_dir / "matrix_scores.csv", output.getvalue())
    lines = [
        "# Model matrix",
        "",
        f"Status: {state['status']}",
        "",
        "| Agent 1 | Agent 2 | Simulation completed / failed | Evaluation completed / failed / excluded |",
        "|---|---|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['agent1_model']} | {row['agent2_model']} | {row['simulation_completed']} / {row['simulation_failed']} | {row['evaluation_completed']} / {row['evaluation_failed']} / {row['evaluation_excluded']} |"
        )
    write_text_atomic(run_dir / "matrix_summary.md", "\n".join(lines) + "\n")
