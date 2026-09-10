"""Evaluation entry points. Model dependencies load only when an episode is evaluated."""

from .cli import (
    PATH_OPTIONS,
    add_arguments,
    automatic_switch,
    resume_overrides,
    validate_arguments,
)
from .runner import (
    check_pair_finished,
    finish_simulation,
    pair_needs_retry,
    pair_settings,
    resume_linked_evaluation,
    run_evaluation,
    validate_linked_evaluation,
)
from .reporting import write_matrix_report

__all__ = [
    "PATH_OPTIONS",
    "add_arguments",
    "automatic_switch",
    "resume_overrides",
    "validate_arguments",
    "check_pair_finished",
    "finish_simulation",
    "pair_needs_retry",
    "pair_settings",
    "resume_linked_evaluation",
    "run_evaluation",
    "validate_linked_evaluation",
    "write_matrix_report",
]
