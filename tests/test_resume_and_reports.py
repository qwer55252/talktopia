import json

import pytest

from talktopia import pipeline, utils
from talktopia.experiment import settings
from talktopia.evaluation.cli import resolve_source_context
from talktopia.evaluation.reporting import write_matrix_report


@pytest.mark.parametrize("mode", ["round-robin", "surface5-full-duplex"])
def test_resume_uses_saved_mode_and_rejects_changed_limits(tmp_path, mode):
    args = pipeline.parse_args(["--resume-run", str(tmp_path), "--episode-limit", "1"])
    saved = {
        "stage": "all",
        "interaction_mode": mode,
        "experiment": settings(mode),
        "max_turns": 12,
        "episode_timeout_s": 120,
        "input_fingerprints": {},
        "tag": "saved-run",
        "database_path": str(tmp_path / "db"),
    }
    config = tmp_path / "run_config.json"
    config.write_text(json.dumps(saved))
    result = utils.restore_run(args, path_fields=(), overrides={})
    assert result.interaction_mode == mode and result.episode_limit == 1
    for field, value in [("max_turns", 99), ("episode_timeout_s", 999)]:
        config.write_text(json.dumps({**saved, field: value}))
        with pytest.raises(ValueError, match="12 budget turns and 120"):
            utils.restore_run(args, path_fields=(), overrides={})


def test_evaluation_inherits_source_mode_and_database(tmp_path, monkeypatch):
    monkeypatch.delenv("TALKTOPIA_DB_DIR", raising=False)
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "interaction_mode": "surface5-full-duplex",
                "database_path": str(tmp_path / "old-db"),
            }
        )
    )
    args = pipeline.parse_args(
        ["--stage", "reevaluate", "--simulation-dir", str(tmp_path)]
    )
    resolve_source_context(args)
    assert args.interaction_mode == "surface5-full-duplex"
    assert args.database_path == str(tmp_path / "old-db")
    args.interaction_mode = "round-robin"
    with pytest.raises(ValueError, match="differs from the source"):
        resolve_source_context(args)


def test_legacy_evaluation_cannot_silently_use_new_database(tmp_path, monkeypatch):
    monkeypatch.delenv("TALKTOPIA_DB_DIR", raising=False)
    episode = tmp_path / "episode.json"
    episode.write_text(json.dumps({"agent_classes": ["CascadedSpeechAgent"] * 2}))
    args = pipeline.parse_args(
        ["--stage", "reevaluate", "--episode-json", str(episode)]
    )
    with pytest.raises(ValueError, match="Source database is unknown"):
        resolve_source_context(args)


def test_report_separates_unique_population_from_matrix_and_pending(tmp_path):
    coverage = {
        "profiles": 40,
        "environments": 90,
        "relationships": 120,
        "combos": 450,
        "combos_per_environment": 5,
        "model_pairs": 16,
        "planned_episodes": 7200,
    }
    state = {
        "status": "partial",
        "interaction_mode": "round-robin",
        "coverage": coverage,
    }
    rows = [
        {
            "agent1_model": "a",
            "agent2_model": "b",
            "simulation_total": 450,
            "simulation_completed": 1,
            "simulation_failed": 0,
        }
        for _ in range(16)
    ]
    write_matrix_report(tmp_path, state, rows)
    report = json.loads((tmp_path / "matrix_summary.json").read_text())
    assert report["coverage"] == coverage
    assert report["simulation_pending"] == 7184
    assert report["status"] == "partial"
    assert report["interaction_mode"] == "round-robin"
    text = (tmp_path / "matrix_summary.md").read_text()
    assert "7200 planned" in text and "7184 pending" in text
