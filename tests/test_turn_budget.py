import asyncio
import json
from types import SimpleNamespace

import pytest

from talktopia import pipeline
from talktopia.full_duplex.actions import DuplexAction, DuplexObservation
from talktopia.full_duplex.config import RuntimeConfig
from talktopia.full_duplex.events import ActionCommitted, EventWriter, read_events
from talktopia.full_duplex.runtime import (
    DuplexRuntime,
    TurnLimitReached,
    select_opening_agent,
)
from sotopia.messages import AgentAction


def test_round_robin_counts_actions_not_environment_steps():
    evaluator = pipeline.env_params(pipeline.parse_args([]))["evaluators"][0]
    messages = []
    for _ in range(11):
        messages.extend(
            [
                ("Alice", AgentAction(action_type="speak", argument="Hello", to=[])),
                ("Bob", AgentAction(action_type="none", argument="", to=[])),
            ]
        )
    # Raw turn number is already beyond 12; eleven real actions are still allowed.
    result = evaluator(23, messages)
    assert not any(value for _, ((name, value), _) in result if name == "terminated")
    messages.append(
        ("Bob", AgentAction(action_type="speak", argument="Goodbye", to=[]))
    )
    result = evaluator(24, messages)
    assert any(value for _, ((name, value), _) in result if name == "terminated")


@pytest.mark.asyncio
async def test_backchannels_keep_event_order_without_consuming_budget(tmp_path):
    class Session:
        calls = 0

        async def commit(self, commit):
            assert commit.expected_turn_number == self.calls
            self.calls += 1
            return SimpleNamespace(
                turn_number=self.calls,
                observations={
                    name: DuplexObservation(
                        observation_id=f"commit-{self.calls}-{name}",
                        last_turn="Received speech",
                        turn_number=self.calls,
                        available_actions=["none", "speak"],
                    )
                    for name in commit.actions
                },
            )

    profiles = [
        SimpleNamespace(first_name=name, last_name="Test") for name in ("Alice", "Bob")
    ]
    resolved = SimpleNamespace(agent_profiles=profiles)
    agents = [
        SimpleNamespace(
            agent_name=f"{p.first_name} Test",
            record_observation=lambda observation: None,
        )
        for p in profiles
    ]
    writer = EventWriter(tmp_path / "events.jsonl", "episode_test")
    session = Session()
    runtime = DuplexRuntime(
        resolved=resolved,
        agents=agents,
        session=session,
        event_writer=writer,
        audio_router=None,
        stereo_writer=None,
        config=RuntimeConfig(),
        model_names=("fake", "fake"),
    )
    runtime._snapshot = SimpleNamespace(turn_number=0, observations={})

    async def commit(kind):
        await runtime._commit_joint(
            actor="Alice Test",
            action=DuplexAction(action_type=kind, argument="Yes"),
            origin="test",
            trigger_event_id=None,
            utterance_id=None,
            metadata={},
        )

    for _ in range(20):
        await commit("backchanneling")
    for _ in range(11):
        await commit("speak")
    assert runtime.state.budget_turns == 11
    assert runtime.state.semantic_turn_number == 31
    with pytest.raises(TurnLimitReached):
        await commit("speak")
    with pytest.raises(TurnLimitReached):
        await commit("backchanneling")
    assert session.calls == 32
    assert runtime.state.budget_turns == 12
    assert runtime.state.action_counts == {"backchanneling": 20, "speak": 12}
    writer.close()
    commits = [
        event
        for event in read_events(tmp_path / "events.jsonl")
        if isinstance(event, ActionCommitted)
    ]
    assert [event.turn_number for event in commits] == list(range(1, 33))
    assert commits[-1].metadata["budget_turns"] == 12
    assert select_opening_agent(resolved, "different_episode") == "Alice Test"


@pytest.mark.asyncio
async def test_round_robin_deadline_cancels_runner_and_preserves_diagnostics(
    tmp_path, monkeypatch
):
    cancelled = asyncio.Event()

    async def stalled(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(pipeline, "run_round_robin_episode", stalled)
    monkeypatch.setattr(pipeline, "EPISODE_TIMEOUT_S", 0.01)
    args = pipeline.parse_args(["--interaction-mode", "round-robin"])
    with pytest.raises(TimeoutError, match="exceeded"):
        await pipeline.run_one_episode(
            SimpleNamespace(turn_number=3), [], args, tmp_path, "episode_0001"
        )
    assert cancelled.is_set()
    diagnostic = json.loads(
        (tmp_path / "simulation/diagnostics/episode_0001.json").read_text()
    )
    assert diagnostic["reason"] == "episode_timeout"
    assert diagnostic["interaction_mode"] == "round-robin"
