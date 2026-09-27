import json

import pytest

from sotopia.database import EpisodeLog, SotopiaDimensions
from talktopia import pipeline
from talktopia.evaluation import evaluator
from talktopia.evaluation.cli import resolve_source_context

# Shared small database and voice fixture; no canonical experiment is launched.
from test_duplex_episode import profiles as profiles


@pytest.mark.asyncio
async def test_both_modes_use_the_same_scoring_prompt_and_settings(
    profiles, tmp_path, monkeypatch
):
    calls = []

    class Judge:
        def __init__(self, **kwargs):
            assert kwargs["response_format_class"] is evaluator.TwoAgentEvaluation

        async def __acall__(self, **kwargs):
            calls.append(kwargs)
            return [
                (agent, ((dimension, 0), f"{agent} recorded evidence for {dimension}"))
                for agent in ("agent_1", "agent_2")
                for dimension in SotopiaDimensions.model_fields
            ]

    monkeypatch.setattr(evaluator, "EpisodeLLMEvaluator", Judge)
    for mode, agent_class in [
        ("round-robin", "CascadedSpeechAgent"),
        ("surface5-full-duplex", "CascadedDuplexAgent"),
    ]:
        source = EpisodeLog(
            environment=profiles["env_id"],
            agents=profiles["agent_ids"],
            agent_classes=[agent_class] * 2,
            tag="source",
            models=["env", "left", "right"],
            messages=[
                [
                    ("Environment", "Alice Test", "Choose a time."),
                    ("Environment", "Bob Test", "Choose a time."),
                ],
                [("Alice Test", "Environment", 'said: "Nine?"')],
                [("Bob Test", "Environment", 'said: "Agreed."')],
            ],
            reasoning="Not evaluated",
            rewards=[0.0, 0.0],
        )
        source_path = tmp_path / f"{mode}.json"
        source_path.write_text(source.model_dump_json())
        args = pipeline.parse_args(
            [
                "--stage",
                "reevaluate",
                "--interaction-mode",
                mode,
                "--episode-json",
                str(source_path),
            ]
        )
        args.reeval_tag = "evaluation-run"
        out = tmp_path / mode
        assert await evaluator.evaluate_episode(args, out) == 0
        summary = json.loads(
            (out / "04_sotopia_eval_reevaluate_existing.json").read_text()
        )
        assert summary["interaction_mode"] == mode and summary["status"] == "completed"
        evaluated = json.loads((out / summary["original"]).read_text())
        assert evaluated["messages"] == json.loads(source.model_dump_json())["messages"]
    assert calls[0] == calls[1]
    assert calls[0]["temperature"] == 0.0 and calls[0]["num_agents"] == 2


@pytest.mark.parametrize(
    "metadata", [{}, {"agent_classes": None}, {"agent_classes": []}]
)
def test_legacy_metadata_allows_explicit_mode_and_database(
    tmp_path, monkeypatch, metadata
):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(metadata))
    monkeypatch.setenv("TALKTOPIA_DB_DIR", str(tmp_path / "old-db"))
    args = pipeline.parse_args(
        [
            "--stage",
            "reevaluate",
            "--episode-json",
            str(path),
            "--interaction-mode",
            "round-robin",
        ]
    )
    resolve_source_context(args)
    assert args.interaction_mode == "round-robin"
    assert args.database_path == str(tmp_path / "old-db")

    class LegacyEpisode:
        agent_classes = metadata.get("agent_classes")

        def render_for_humans(self):
            return ["profiles"], ["saved ASR"]

    assert evaluator.round_robin_history(LegacyEpisode()) == (
        ["profiles"],
        ["saved ASR"],
    )
