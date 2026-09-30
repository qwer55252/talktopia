import copy
import hashlib
import json

import pytest

from talktopia import experiment, pipeline


@pytest.fixture
def population(monkeypatch):
    source = {
        "AgentProfile": [
            {"pk": f"a{i:02}", "first_name": f"Agent{i}", "last_name": "Test"}
            for i in range(40)
        ],
        "EnvironmentProfile": [
            {"pk": f"e{i:02}", "scenario": "A task"} for i in range(90)
        ],
        "RelationshipProfile": [{"pk": f"r{i:03}"} for i in range(120)],
        "EnvAgentComboStorage": [
            {
                "pk": f"c{i:03}_{j}",
                "env_id": f"e{i:02}",
                "agent_ids": [f"a{(i + j) % 40:02}", f"a{(i + j + 1) % 40:02}"],
            }
            for i in range(90)
            for j in range(5)
        ],
    }
    lock = {key: {"sha256": "test-hash"} for key in source}
    monkeypatch.setattr(experiment, "canonical_dataset", lambda: (source, lock))
    records = [
        {
            "episode_id": f"episode_{i:04}",
            "combo_id": row["pk"],
            "env_id": row["env_id"],
            "agent_ids": row["agent_ids"],
        }
        for i, row in enumerate(source["EnvAgentComboStorage"], 1)
    ]
    return source, records


def test_complete_population_and_matrix_size(population):
    _, records = population
    coverage = experiment.validate_manifest(records)
    assert coverage == {
        "profiles": 40,
        "environments": 90,
        "relationships": 120,
        "combos": 450,
        "combos_per_environment": 5,
    }
    assert coverage["combos"] * 16 == 7200


@pytest.mark.asyncio
async def test_selected_episodes_preserve_full_population_and_resume_unfinished(
    population, tmp_path, monkeypatch
):
    _, records = population
    selected = ["episode_0021", "episode_0016", "episode_0011", "episode_0006"]
    argv = ["--episode-limit", "2"]
    for episode_id in selected:
        argv.extend(["--episode-id", episode_id])
    args = pipeline.parse_args(argv)
    args.tag = "selected-test"
    calls = []
    monkeypatch.setattr(pipeline, "result_artifacts", lambda *args: {"test": "hash"})

    async def run(record, artifact_dir, result_path):
        calls.append(record["episode_id"])
        return {**record, "status": "completed"}

    assert await pipeline.run_simulation_batch(records, args, tmp_path, run) == 0
    assert calls == ["episode_0006", "episode_0011"]
    summary = json.loads((tmp_path / "03_simulation.json").read_text())
    assert (summary["total"], summary["completed"], summary["pending"]) == (450, 2, 448)
    assert summary["coverage"] == experiment.validate_manifest(records)

    args.resume_run = tmp_path
    args.episode_limit = 4
    assert await pipeline.run_simulation_batch(records, args, tmp_path, run) == 0
    assert calls == sorted(selected)
    assert await pipeline.run_simulation_batch(records, args, tmp_path, run) == 0
    assert calls == sorted(selected)  # No completed episode is run again.
    summary = json.loads((tmp_path / "03_simulation.json").read_text())
    assert (summary["total"], summary["completed"], summary["pending"]) == (450, 4, 446)
    assert summary["status"] == "partial"
    assert [row["episode_id"] for row in summary["episodes"]] == [
        row["episode_id"] for row in records
    ]
    assert all(
        row["status"] == "pending" and row["attempt_history"] == []
        for row in summary["episodes"] if row["episode_id"] not in selected
    )


def test_invalid_episode_selection_fails_before_server_startup(
    population, tmp_path, monkeypatch
):
    import talktopia.task_space

    _, records = population
    monkeypatch.setattr(talktopia.task_space, "configure_database", lambda _: tmp_path)
    monkeypatch.setattr(pipeline, "validate_dataset", lambda _: {})
    monkeypatch.setattr(pipeline, "stage_1_sample_env_profiles", lambda *args: [])
    monkeypatch.setattr(pipeline, "stage_2_sample_characters", lambda *args: records)
    args = pipeline.parse_args(["--episode-id", "episode_0006"])
    assert pipeline.preflight(args) == tmp_path
    args.episode_ids.append("episode_9999")
    with pytest.raises(ValueError, match="Unknown simulation episode IDs"):
        pipeline.preflight(args)
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--episode-id", "episode_0006", "--episode-id", "episode_0006"])
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--stage", "reevaluate", "--episode-id", "episode_0006"])


def test_repeated_four_combos_cannot_impersonate_full_population(population):
    _, records = population
    repeated = [
        {**records[i % 4], "episode_id": f"episode_{i + 1:04}"} for i in range(450)
    ]
    with pytest.raises(ValueError, match="duplicate"):
        experiment.validate_manifest(repeated)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "roles", "unknown", "order", "episode_id"]
)
def test_invalid_populations_fail(population, change):
    _, original = population
    records = copy.deepcopy(original)
    if change == "missing":
        records.pop()
    elif change == "extra":
        records.append(records[0])
    elif change == "roles":
        records[0]["agent_ids"].reverse()
    elif change == "unknown":
        records[0]["combo_id"] = "unknown"
    elif change == "order":
        records.reverse()
    else:
        records[0]["episode_id"] = "episode_9999"
    with pytest.raises(ValueError):
        experiment.validate_manifest(records)


def test_database_with_same_count_but_modified_profile_is_rejected(
    population, tmp_path, monkeypatch
):
    source, _ = population
    import talktopia.speech_agent

    monkeypatch.setattr(talktopia.speech_agent, "load_voice_registry", lambda path: {})
    for name, rows in source.items():
        (tmp_path / name).mkdir()
        for row in rows:
            (tmp_path / name / f"{row['pk']}.json").write_text(json.dumps(row))
    experiment.validate_dataset(tmp_path)
    row = {**source["AgentProfile"][0], "first_name": "Changed"}
    (tmp_path / "AgentProfile" / f"{row['pk']}.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="source fields changed"):
        experiment.validate_dataset(tmp_path)


def test_source_hash_checked_before_reading_records(tmp_path, monkeypatch):
    raw = b'{"pk":"a"}\n'
    path = tmp_path / "agents.jsonl"
    path.write_bytes(raw)
    (tmp_path / "dataset.lock.json").write_text(
        json.dumps(
            {
                "AgentProfile": {
                    "file": path.name,
                    "count": 1,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            }
        )
    )
    monkeypatch.setattr(experiment, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("TALKTOPIA_GEMINILIGHT_DATA_DIR", str(tmp_path))
    assert experiment.canonical_dataset()[0]["AgentProfile"] == [{"pk": "a"}]
    path.write_bytes(b'{"pk":"b"}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        experiment.canonical_dataset()


def test_mode_defaults_and_fixed_limits():
    args = pipeline.parse_args([])
    assert (args.interaction_mode, args.max_turns, args.episode_timeout_s) == (
        "round-robin",
        12,
        120,
    )
    args = pipeline.parse_args(["--interaction-mode", "surface5-full-duplex"])
    assert (args.max_turns, args.episode_timeout_s) == (12, 120)


@pytest.mark.parametrize(
    "args",
    [
        ["--tag", "test"],
        ["--reeval-tag", "test"],
        ["--num-envs", "4"],
        ["--pairs-per-env", "1"],
        ["--env-id", "e1"],
        ["--max-turns", "20"],
        ["--episode-timeout-s", "150"],
    ],
)
def test_removed_overrides_cannot_change_population_or_limits(args):
    with pytest.raises(SystemExit):
        pipeline.parse_args(args)
