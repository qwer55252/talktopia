from __future__ import annotations

import hashlib
import json
import wave
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from demo.app import create_app
from demo.evaluations import DIMENSIONS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def write_wav(path, seconds, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((channels, 2, 1000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * channels * round(seconds * 1000))


def write_speech(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved(tmp_path):
    sim, db = tmp_path / "sim", tmp_path / "db"
    original = {
        "environment": "env-1", "agents": ["alice", "bob"],
        "agent_classes": ["CascadedSpeechAgent", "CascadedSpeechAgent"],
        "models": ["environment-model", "actual-model-a", "actual-model-b"],
        "messages": [
            [
                ["Environment", "Alice", "Alice's original context"],
                ["Environment", "Bob", "Bob's original context"],
                ["Alice", "Environment", '[private to [\'Bob\']]  said: "ASR one"'],
                ["Bob", "Environment", " did nothing"],
            ],
            [["Bob", "Environment", "[non-verbal communication] nods"]],
            [["Bob", "Environment", '[private to [\'Alice\']]  said: "ASR two"']],
        ],
    }
    source_path = sim / "simulation/original/episode_0001.json"
    write_json(source_path, original)
    profiles = {
        "AgentProfile/alice.json": {"pk": "alice", "first_name": "Alice", "secret": "A secret"},
        "AgentProfile/bob.json": {"pk": "bob", "first_name": "Bob"},
        "EnvironmentProfile/env-1.json": {"pk": "env-1", "relationship": 4, "agent_goals": ["A", "B"]},
        "RelationshipProfile/wrong.json": {"pk": "wrong", "agent_1_id": "alice", "agent_2_id": "bob", "relationship": 0},
        "RelationshipProfile/right.json": {"pk": "right", "agent_1_id": "bob", "agent_2_id": "alice", "relationship": 4},
    }
    for name, value in profiles.items():
        write_json(db / name, value)
    write_json(sim / "run_config.json", {
        "input_fingerprints": {str(db / name): digest(db / name) for name in profiles}
    })
    row = {
        "episode_id": "episode_0001", "env_id": "env-1", "agent_ids": ["alice", "bob"],
        "agent_names": ["Alice", "Bob"], "codename": "conversation",
        "status": "completed", "error": "Old attempt failed",
        "original": "simulation/original/episode_0001.json",
        "readable": "simulation/readable/episode_0001.md",
        "speech": "simulation/speech/episode_0001.jsonl",
        "conversation_audio": "simulation/audio/episode_0001/conversation.wav",
        "models": {"agent1": "incorrect-manifest-model", "agent2": "incorrect"},
    }
    manifest = [
        {key: value for key, value in row.items() if key not in ("status", "error")},
        {"episode_id": "episode_0002", "env_id": "env-1", "agent_ids": ["alice", "bob"]},
        {"episode_id": "episode_0003", "env_id": "env-1", "agent_ids": ["alice", "bob"]},
    ]
    write_json(sim / "02_sampled_characters.json", manifest)
    summary = {"episodes": [row, {**manifest[2], "status": "failed", "error": "Timeout"}]}
    write_json(sim / "03_simulation.json", summary)
    speech = [
        {"turn": 1, "speaker": "Alice", "status": "completed", "asr_text": "ASR one",
         "tts_text": "Never substitute TTS", "wav_path": "simulation/audio/episode_0001/a.wav"},
        {"turn": 2, "speaker": "Alice", "status": "tts_skipped", "asr_text": None, "wav_path": None},
        {"turn": 3, "speaker": "Bob", "status": "completed", "asr_text": "ASR two",
         "wav_path": "simulation/audio/episode_0001/b.wav"},
    ]
    write_speech(sim / row["speech"], speech)
    write_wav(sim / speech[0]["wav_path"], 1)
    write_wav(sim / speech[2]["wav_path"], 2)
    write_wav(sim / row["conversation_audio"], 3.3, channels=2)
    (sim / row["readable"]).parent.mkdir(parents=True)
    (sim / row["readable"]).write_text("# Original conversation\n\nASR one\n")
    evaluations = []
    for index in (1, 2):
        directory = tmp_path / f"evaluation-{index}"
        scores = {name: 1.0 * index for name in DIMENSIONS}
        scores["relationship"] = -1
        write_json(directory / "evaluation/original/episode_0001.json", {
            **original, "rewards": [[index, scores], [index, scores]],
        })
        eval_row = {
            **row, "source_episode": "/previous/machine/episode_0001.json",
            "source_sha256": digest(source_path), "evaluator_model": f"evaluator-{index}",
            "original": "evaluation/original/episode_0001.json",
            "readable": "evaluation/readable/episode_0001.md",
        }
        write_json(directory / "evaluation_manifest.json", [eval_row])
        write_json(directory / "04_sotopia_eval_reevaluate_existing.json", {"episodes": [eval_row]})
        report = directory / eval_row["readable"]
        report.parent.mkdir(parents=True)
        report.write_text(
            "# Evaluation\n\n| Dimension | Alice | Bob |\n|---|---:|---:|\n| goal | 1 | 2 |\n\n"
            "<script>window.executed=true</script>\n\n<naturalness> Text must remain visible.\n\n"
            "[bad](javascript:alert(1))\n"
        )
        evaluations.append(directory)
    config = tmp_path / "config.json"
    config_value = {"runs": [{"id": "test", "simulation_dir": "sim",
                              "evaluation_dirs": [str(p) for p in evaluations]}]}
    write_json(config, config_value)
    with TestClient(create_app(config)) as client:
        yield SimpleNamespace(
            client=client, config=config, config_value=config_value, sim=sim, db=db,
            source=source_path, original=original, row=row, summary=summary,
            evaluations=evaluations, speech=speech, base="/api/runs/test/episodes/episode_0001",
        )


def test_lists_all_statuses_and_loads_only_requested_episode(saved):
    data = saved.client.get("/api/runs/test/episodes").json()
    assert [row["status"] for row in data["episodes"]] == ["completed", "pending", "failed"]
    assert saved.client.get("/api/runs/test/episodes/episode_9999").status_code == 404
    detail = saved.client.get(saved.base).json()
    assert detail["models"] == ["actual-model-a", "actual-model-b"]
    assert detail["error"] is None
    assert detail["profiles"]["relationship"]["pk"] == "right"
    assert detail["profiles"]["environment"]["agent_goals"] == ["A", "B"]
    assert [row["association"] for row in detail["evaluations"]] == ["matched", "matched"]
    assert [row["scores"][0]["goal"] for row in detail["evaluations"]] == [1, 2]
    assert detail["evaluations"][0]["scores"][0]["relationship"] == -1


def test_round_robin_private_speech_and_skipped_tts_timing(saved):
    playback = saved.client.get(saved.base).json()["playback"]
    assert playback["duration"] == pytest.approx(3.3)
    assert [(row["text"], row["start"]) for row in playback["entries"]] == [
        ("ASR one", 0), ("[non-verbal communication] nods", None), ("ASR two", 1.3),
    ]
    assert not playback["warnings"]


@pytest.mark.parametrize("change", ["missing_chunk", "wrong_duration", "partial_jsonl"])
def test_uncertain_audio_timing_preserves_audio_and_text(saved, change):
    if change == "missing_chunk":
        (saved.sim / saved.speech[0]["wav_path"]).unlink()
    elif change == "wrong_duration":
        write_wav(saved.sim / saved.row["conversation_audio"], 4.0, channels=2)
    else:
        with (saved.sim / saved.row["speech"]).open("a") as output:
            output.write('{"turn":')
    detail = saved.client.get(saved.base).json()
    assert detail["playback"]["audio_available"]
    assert all(row["start"] is None for row in detail["playback"]["entries"])
    assert detail["playback"]["warnings"]
    assert detail["playback"]["entries"][0]["text"] == "ASR one"


def test_retry_paths_are_authoritative(saved):
    retry = "attempts/simulation/episode_0001/2/"
    row = saved.summary["episodes"][0]
    for key in ("original", "readable", "speech", "conversation_audio"):
        previous = saved.sim / row[key]
        destination = saved.sim / (retry + row[key])
        destination.parent.mkdir(parents=True, exist_ok=True)
        previous.rename(destination)
        row[key] = retry + row[key]
    for item in saved.speech:
        if item["wav_path"]:
            previous = saved.sim / item["wav_path"]
            destination = saved.sim / (retry + item["wav_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            previous.rename(destination)
            item["wav_path"] = retry + item["wav_path"]
    write_speech(saved.sim / row["speech"], saved.speech)
    write_json(saved.sim / "03_simulation.json", saved.summary)
    detail = saved.client.get(saved.base).json()
    assert detail["error"] is None
    assert detail["evaluations"][0]["association"] == "matched"
    assert detail["playback"]["entries"][2]["start"] == pytest.approx(1.3)
    assert saved.client.get(saved.base + "/reports/simulation").status_code == 200


@pytest.mark.parametrize("change", ["hash", "agents", "environment", "missing_hash", "events"])
def test_evaluation_identity_is_verified_before_scores(saved, change):
    path = saved.evaluations[0] / "04_sotopia_eval_reevaluate_existing.json"
    data = json.loads(path.read_text())
    row = data["episodes"][0]
    if change == "hash":
        row["source_sha256"] = "0" * 64
    elif change == "agents":
        row["agent_ids"] = ["bob", "alice"]
    elif change == "environment":
        row["env_id"] = "different"
    elif change == "missing_hash":
        row["source_sha256"] = None
    else:
        row["source_events_sha256"] = "0" * 64
    write_json(path, data)
    bad, good = saved.client.get(saved.base).json()["evaluations"]
    assert bad["scores"] is None
    assert bad["association"] != "matched"
    assert bad["report_available"]
    assert good["scores"] is not None


@pytest.mark.parametrize("status", ["pending", "failed", "excluded"])
def test_noncompleted_evaluations_are_not_zero_scores(saved, status):
    path = saved.evaluations[0] / "04_sotopia_eval_reevaluate_existing.json"
    data = json.loads(path.read_text())
    data["episodes"][0]["status"] = status
    write_json(path, data)
    row = saved.client.get(saved.base).json()["evaluations"][0]
    assert row["status"] == status
    assert row["scores"] is None


def test_missing_and_changed_profiles_fall_back_to_initial_context(saved):
    (saved.db / "AgentProfile/alice.json").write_text('{"pk":"alice","secret":"changed"}')
    (saved.db / "EnvironmentProfile/env-1.json").unlink()
    detail = saved.client.get(saved.base).json()
    assert detail["profiles"]["agents"][0] is None
    assert detail["profiles"]["environment"] is None
    assert detail["profiles"]["relationship"] is None
    assert detail["profiles"]["warnings"]
    assert detail["perspectives"][0]["text"] == "Alice's original context"


def test_profile_directory_can_move_without_rewriting_results(saved):
    moved = saved.db.with_name("relocated")
    saved.db.rename(moved)
    saved.config_value["runs"][0]["profiles_dir"] = str(moved)
    write_json(saved.config, saved.config_value)
    detail = saved.client.get(saved.base).json()
    assert detail["profiles"]["agents"][0]["first_name"] == "Alice"
    assert detail["profiles"]["relationship"]["pk"] == "right"


def test_config_refresh_adds_evaluator_without_restart(saved):
    saved.config_value["runs"][0]["evaluation_dirs"] = [str(saved.evaluations[0])]
    write_json(saved.config, saved.config_value)
    assert len(saved.client.get(saved.base).json()["evaluations"]) == 1
    saved.config_value["runs"][0]["evaluation_dirs"].append(str(saved.evaluations[1]))
    write_json(saved.config, saved.config_value)
    assert len(saved.client.get(saved.base).json()["evaluations"]) == 2


def test_same_episode_id_in_another_run_does_not_share_results(saved):
    other = saved.sim.with_name("other")
    other.mkdir()
    write_json(other / "02_sampled_characters.json", [{"episode_id": "episode_0001"}])
    write_json(other / "03_simulation.json", {"episodes": []})
    saved.config_value["runs"].append({"id": "other", "simulation_dir": str(other)})
    write_json(saved.config, saved.config_value)
    detail = saved.client.get("/api/runs/other/episodes/episode_0001").json()
    assert detail["status"] == "pending"
    assert detail["evaluations"] == []
    assert detail["models"] == []


def test_in_progress_json_and_missing_artifacts_do_not_break_listing(saved):
    (saved.sim / "03_simulation.json").write_text('{"episodes":[')
    listing = saved.client.get("/api/runs/test/episodes").json()
    assert len(listing["episodes"]) == 3
    assert listing["warnings"]
    saved.source.unlink()
    detail = saved.client.get(saved.base).json()
    assert detail["evaluations"][0]["scores"] is None
    assert detail["playback"]["entries"][0]["text"] == "ASR one"


def test_audio_range_and_head(saved):
    response = saved.client.get(saved.base + "/audio", headers={"Range": "bytes=0-43"})
    assert response.status_code == 206
    assert response.content[:4] == b"RIFF"
    assert len(response.content) == 44
    assert response.headers["content-range"].startswith("bytes 0-43/")
    assert saved.client.head(saved.base + "/audio").content == b""
    assert saved.client.get(saved.base + "/audio", headers={"Range": "bytes=999999-"}).status_code == 416


@pytest.mark.parametrize("attack", ["parent", "absolute", "symlink"])
def test_audio_cannot_escape_registered_directory(saved, attack):
    outside = saved.sim.parent / "private.wav"
    outside.write_bytes(b"secret")
    if attack == "parent":
        saved.row["conversation_audio"] = "../private.wav"
    elif attack == "absolute":
        saved.row["conversation_audio"] = str(outside)
    else:
        link = saved.sim / "linked.wav"
        link.symlink_to(outside)
        saved.row["conversation_audio"] = "linked.wav"
    write_json(saved.sim / "03_simulation.json", saved.summary)
    response = saved.client.get(saved.base + "/audio")
    assert response.status_code == 422
    assert b"secret" not in response.content


def test_markdown_keeps_tables_and_literal_tags_without_execution(saved):
    evaluation = saved.client.get(saved.base).json()["evaluations"][0]
    response = saved.client.get(saved.base + "/reports/" + evaluation["id"])
    rendered = response.json()["html"]
    assert "<table>" in rendered
    assert "style=" not in rendered
    assert 'class="align-right"' in rendered
    assert "<script>" not in rendered
    assert "&lt;naturalness&gt;" in rendered
    assert 'href="javascript:' not in rendered
    assert "<script>" in response.json()["markdown"]
    assert "script-src 'self'" in response.headers["content-security-policy"]


def test_duplex_timing_overlaps_and_nonverbal_actions_use_events(saved):
    speech = [
        {"utterance_id": "a", "speaker": "Alice", "asr_text": "Opening", "start_ms": 0,
         "end_ms": 2000, "action_type": "speak", "commit_id": "later", "wav_path": None},
        {"utterance_id": "b", "speaker": "Bob", "asr_text": "Yes", "start_ms": 500,
         "end_ms": 900, "action_type": "backchanneling", "commit_id": "earlier", "wav_path": None},
    ]
    write_speech(saved.sim / saved.row["speech"], list(reversed(speech)))
    saved.row.update(interaction_mode="surface5-full-duplex", events="simulation/events/episode_0001.jsonl")
    write_speech(saved.sim / saved.row["events"], [
        {"event_type": "action_committed", "timestamp_ms": 1500,
         "actions": {"Bob": {"action_type": "non-verbal communication", "argument": "nods"}}}
    ])
    write_json(saved.sim / "03_simulation.json", saved.summary)
    entries = saved.client.get(saved.base).json()["playback"]["entries"]
    assert [(row["text"], row["start"], row["end"]) for row in entries] == [
        ("Opening", 0, 2), ("Yes", .5, .9), ("nods", 1.5, 1.5),
    ]
    assert all(row["committed"] for row in entries)


def test_http_reads_do_not_change_artifacts(saved):
    paths = [p for p in saved.sim.parent.rglob("*") if p.is_file()]
    before = {p: digest(p) for p in paths}
    saved.client.get("/api/runs")
    saved.client.get("/api/runs/test/episodes")
    saved.client.get(saved.base)
    saved.client.get(saved.base + "/audio")
    saved.client.get(saved.base + "/reports/simulation")
    assert {p: digest(p) for p in paths} == before


def test_exclusions_in_summary_are_displayed_with_their_reason(saved):
    directory = saved.evaluations[0]
    write_json(directory / "evaluation_manifest.json", [])
    write_json(directory / "04_sotopia_eval_reevaluate_existing.json", {
        "episodes": [],
        "excluded_episodes": [{
            "episode_id": "episode_0001", "env_id": "env-1",
            "agent_ids": ["alice", "bob"], "reason": "no_interaction",
        }],
    })
    evaluation = saved.client.get(saved.base).json()["evaluations"][0]
    assert evaluation["status"] == "excluded"
    assert evaluation["scores"] is None
    assert "no_interaction" in evaluation["warnings"]


def test_duplex_missing_speech_keeps_audible_original_messages(saved):
    saved.row.update(interaction_mode="surface5-full-duplex", events="simulation/events/episode_0001.jsonl")
    (saved.sim / saved.row["speech"]).unlink()
    write_speech(saved.sim / saved.row["events"], [
        {"event_type": "action_committed", "timestamp_ms": 1500,
         "actions": {"Bob": {"action_type": "non-verbal communication", "argument": "nods"}}}
    ])
    write_json(saved.sim / "03_simulation.json", saved.summary)
    entries = saved.client.get(saved.base).json()["playback"]["entries"]
    assert len(entries) == 3
    assert {row["text"] for row in entries} == {"ASR one", "ASR two", "nods"}
    assert all(row["start"] is None for row in entries if row["action"] == "speak")
