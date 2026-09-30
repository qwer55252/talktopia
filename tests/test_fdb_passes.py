"""Keep real pass actions distinct from simultaneous peer placeholders."""

from types import SimpleNamespace

import pytest
from sotopia.agents.llm_agent import Agents, LLMAgent
from sotopia.database import EnvironmentProfile
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
from sotopia.envs.parallel import ParallelSotopiaEnv
from test_duplex_episode import profiles as profiles  # noqa: PLC0414 -- fixture

from talktopia.evaluation.temporal import timed_messages
from talktopia.full_duplex.actions import DuplexAction, DuplexObservation
from talktopia.full_duplex.events import ActionCommitted
from talktopia.full_duplex.rendering import render_sotopia_messages
from talktopia.full_duplex.sotopia_adapter import (
    DuplexSotopiaEnv,
    ResolvedEpisode,
    SemanticCommit,
    SotopiaSession,
)
from talktopia.speech_agent import AgentProfile


def test_initial_context_matches_round_robin_without_exposing_peer_private_data(
    profiles,
):
    env_profile = EnvironmentProfile.get(profiles["env_id"])
    agent_profiles = [AgentProfile.get(pk) for pk in profiles["agent_ids"]]
    agents = Agents(
        {
            f"{profile.first_name} {profile.last_name}": LLMAgent(agent_profile=profile)
            for profile in agent_profiles
        }
    )
    surface5 = DuplexSotopiaEnv(env_profile)
    round_robin = ParallelSotopiaEnv(
        env_profile=env_profile, action_order="round-robin"
    )
    try:
        duplex_observations = surface5.reset(agents=agents)
        baseline_observations = round_robin.reset(agents=agents)
        assert surface5.background.model_dump() == round_robin.background.model_dump()
        for index, name in enumerate(agents):
            observation = duplex_observations[name]
            baseline = baseline_observations[name]
            assert observation.last_turn == baseline.last_turn
            assert observation.to_natural_language() == baseline.to_natural_language()
            peer = list(agents)[1 - index]
            assert env_profile.scenario in observation.last_turn
            assert env_profile.agent_goals[index] in observation.last_turn
            assert agent_profiles[index].secret in observation.last_turn
            assert f"{peer}'s background: Unknown" in observation.last_turn
            assert f"{peer}'s goal: Unknown" in observation.last_turn
            assert env_profile.agent_goals[1 - index] not in observation.last_turn
            assert agent_profiles[1 - index].secret not in observation.last_turn
    finally:
        surface5.close()
        round_robin.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_types", "expected"),
    [
        (["none", "none", "none"], [False, False, True]),
        (
            ["none", "none", "speak", "none", "none", "none"],
            [False, False, False, False, False, True],
        ),
    ],
)
async def test_actual_pass_staleness_matches_round_robin(
    profiles, action_types, expected
):
    agent_profiles = tuple(AgentProfile.get(pk) for pk in profiles["agent_ids"])
    resolved = ResolvedEpisode(
        combo_pk="combo-pass",
        env_profile=EnvironmentProfile.get(profiles["env_id"]),
        agent_profiles=agent_profiles,
    )
    agents = tuple(LLMAgent(agent_profile=profile) for profile in agent_profiles)
    session = SotopiaSession(resolved)
    with pytest.raises(RuntimeError, match="not open"):
        session.is_stale([])
    session.open(agents)
    round_robin = ParallelSotopiaEnv(
        env_profile=resolved.env_profile,
        action_order="round-robin",
        evaluators=[RuleBasedTerminatedEvaluator(max_stale_turn=2)],
    )
    round_robin.reset(agents=Agents({agent.agent_name: agent for agent in agents}))
    names = [agent.agent_name for agent in agents]
    actions = []
    assert session.is_stale(actions) is False
    try:
        for index, (action_type, should_stop) in enumerate(zip(action_types, expected)):
            actor = names[index % 2]
            action = DuplexAction(
                action_type=action_type,
                argument="Let us consider another time."
                if action_type == "speak"
                else "",
            )
            joint = {name: DuplexAction(action_type="none") for name in names}
            joint[actor] = action
            await session.commit(
                SemanticCommit(
                    actions=joint,
                    expected_turn_number=index,
                    timestamp_ms=index,
                    origin="agent",
                    utterance_ids={name: None for name in names},
                    metadata={"actor": actor},
                )
            )
            actions.append((actor, action))
            _, _, terminated, _, _ = await round_robin.astep(joint)
            assert session.is_stale(actions) is should_stop
            assert all(value is should_stop for value in terminated.values())
            # The simultaneous environment records placeholders, but this helper
            # must use only the actual action list supplied by the runtime.
            fd_actions = [
                speaker
                for speaker, _ in session._environment.inbox
                if speaker != "Environment"
            ]
            rr_actions = [
                speaker for speaker, _ in round_robin.inbox if speaker != "Environment"
            ]
            assert len(fd_actions) == 2 * (index + 1)
            assert len(rr_actions) == index + 1
    finally:
        session.close()
        round_robin.close()


def pass_commit(index, actor=None, **metadata):
    return ActionCommitted(
        event_id=f"event-{index}",
        episode_id="episode-pass",
        sequence=index,
        timestamp_ms=1000 * index,
        commit_id=f"commit-{index}",
        turn_number=index,
        actions={name: DuplexAction(action_type="none") for name in ("Alice", "Bob")},
        observations_after={},
        origin="agent",
        metadata={**metadata, **({"actor": actor} if actor is not None else {})},
    )


def test_actual_pass_is_preserved_in_both_histories_without_peer_placeholder():
    initial = {
        "Alice": DuplexObservation(
            observation_id="initial",
            last_turn="Arrange a meeting.",
            turn_number=0,
            available_actions=["none", "speak"],
        )
    }
    commits = [
        pass_commit(1, "Alice"),
        pass_commit(2, "Bob", generation_failure=True),
        pass_commit(3),  # Historical all-none commits lack an actual actor.
    ]
    messages = render_sotopia_messages(initial, commits, [])
    assert messages[1:] == [
        [("Alice", "Environment", " did nothing")],
        [("Bob", "Environment", " did nothing")],
    ]
    assert timed_messages(SimpleNamespace(messages=messages), commits, [], []) == [
        messages[0],
        [("Alice", "Environment", "[00:01.000]  did nothing")],
        [("Bob", "Environment", "[00:02.000]  did nothing")],
    ]


def test_legacy_non_none_action_remains_visible_and_still_requires_audible_asr():
    commit = pass_commit(1).model_copy(
        update={
            "actions": {
                "Alice": DuplexAction(
                    action_type="action", argument="Checks the calendar."
                ),
                "Bob": DuplexAction(action_type="none"),
            }
        }
    )
    messages = render_sotopia_messages({}, [commit], [])
    assert messages[1] == [("Alice", "Environment", " [action] Checks the calendar.")]
    assert timed_messages(SimpleNamespace(messages=messages), [commit], [], [])[1] == [
        ("Alice", "Environment", "[00:01.000]  [action] Checks the calendar.")
    ]
    audible = commit.model_copy(
        update={
            "actions": {
                "Alice": DuplexAction(action_type="speak", argument="Unheard words."),
                "Bob": DuplexAction(action_type="none"),
            }
        }
    )
    with pytest.raises(ValueError, match="has no transcript entry"):
        render_sotopia_messages({}, [audible], [])
