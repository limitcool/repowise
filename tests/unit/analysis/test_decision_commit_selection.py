"""A commit-mined decision binds the files the model chose, not the commit's.

The defect these pin: ``pr`` and ``git_archaeology`` read one decision out of
one commit body and then took that commit's *whole* file list, because the
model was never asked which files the decision was about. A commit that
bundles three unrelated changes therefore produced three decisions each
claiming all of its files, and ``get_why`` on any of them answered with the
other two.

Measured out of sample on 29 records drawn from the dev store: the stored
commit lists are 27% on topic with 18% outright noise, and the same records
with the files chosen by a model reading the commit body are 93% on topic with
none noise, losing no record that had a real answer. Evidence in
``local-stash/decision-layer-research/capture-2026-09-19/RESULTS.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from repowise.core.analysis.decisions.extractor import DecisionExtractor
from repowise.core.analysis.decisions.scope import (
    SCOPE_BASIS_FOOTPRINT,
    SCOPE_BASIS_SELECTED,
    selected_scope_files,
)

#: One commit, two parts. ``_SUBJECT`` is what the decision under test is
#: about; the rest is what the same commit happened to touch.
_SUBJECT = "packages/core/src/repowise/core/hook.py"
_BYSTANDERS = [f"packages/core/src/repowise/core/mod{i:02d}.py" for i in range(8)]
_COMMIT_FILES = sorted([_SUBJECT, *_BYSTANDERS])

_SHA = "9a0b27a7"


def _meta_map(files: list[str] = _COMMIT_FILES) -> dict[str, dict]:
    """A ``_git_meta_map`` whose inversion yields one commit over *files*."""
    commit = {
        "sha": _SHA,
        "message": "perf: stop importing the workspace stack from the hook path",
        "subject": "perf: stop importing the workspace stack from the hook path",
        # Shaped so both miners accept it: the PR miner wants a PR-ish body
        # marker plus a decision signal, git archaeology wants the signal.
        "body": (
            "## Why\n\n"
            "The hook imported core.workspace.config only to reach "
            "find_workspace_root, which initialized the extractor stack. We "
            "chose instead to resolve CLI command modules lazily, because the "
            "import cost was charged to every invocation."
        ),
        "pr_number": 2439,
        "author": "someone",
        "date": "2026-09-01",
    }
    return {f: {"significant_commits_json": json.dumps([commit])} for f in files}


class _Provider:
    """Returns one canned decision payload, and records what it was asked."""

    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload
        self.prompts: list[str] = []

    async def generate(self, system, prompt, **kwargs):  # noqa: ANN001
        self.prompts.append(prompt)
        return SimpleNamespace(content=json.dumps(self._payload))


def _extractor(tmp_path: Path, payload: list[dict]) -> DecisionExtractor:
    provider = _Provider(payload)
    ex = DecisionExtractor(
        repo_path=tmp_path, provider=provider, git_meta_map=_meta_map()
    )
    ex.test_provider = provider  # type: ignore[attr-defined]
    return ex


def _payload(**extra) -> list[dict]:
    return [
        {
            "commit_sha": _SHA,
            "title": "Avoid the workspace dependency graph in agent hooks",
            "decision": "Drop the module-level import from the hook path.",
            "rationale": "It initialized the whole extractor stack.",
            **extra,
        }
    ]


# --- the validator ---------------------------------------------------------


def test_selection_keeps_only_files_the_commit_touched():
    """A path outside the commit's list is a path the model invented.

    Intersecting is the whole of the validation: the miner has no other way to
    tell a real path from a plausible one, and the model is being shown the
    list it is meant to be choosing from.
    """
    kept = selected_scope_files([_SUBJECT, "packages/core/src/invented.py"], _COMMIT_FILES)
    assert kept == [_SUBJECT]


def test_selection_normalises_separators_before_comparing():
    """A Windows-shaped answer still matches a POSIX-shaped commit list."""
    assert selected_scope_files(
        ["packages\\core\\src\\repowise\\core\\hook.py"], _COMMIT_FILES
    ) == [_SUBJECT]


def test_selection_of_nothing_is_empty_rather_than_everything():
    assert selected_scope_files([], _COMMIT_FILES) == []


# --- the pr miner ----------------------------------------------------------


async def test_pr_binds_the_selected_file_and_not_the_commit(tmp_path):
    ex = _extractor(tmp_path, _payload(affected_files=[_SUBJECT]))

    (decision,) = await ex.mine_pr_bodies()

    assert decision.affected_files == [_SUBJECT]
    assert decision.scope_basis == SCOPE_BASIS_SELECTED
    for bystander in _BYSTANDERS:
        assert bystander not in decision.affected_files


async def test_pr_drops_a_path_the_commit_never_touched(tmp_path):
    ex = _extractor(
        tmp_path, _payload(affected_files=[_SUBJECT, "packages/core/src/ghost.py"])
    )

    (decision,) = await ex.mine_pr_bodies()

    assert decision.affected_files == [_SUBJECT]


async def test_pr_binds_nothing_when_the_model_selects_nothing(tmp_path):
    """An empty answer must not fall back to the commit's whole list.

    Falling back would reinstate exactly what this replaces, on the records it
    helps most: roughly one in six, every one of them measured as a record
    whose subject never reached the commit's file list at all.
    """
    ex = _extractor(tmp_path, _payload(affected_files=[]))

    (decision,) = await ex.mine_pr_bodies()

    assert decision.affected_files == []
    assert decision.scope_basis == SCOPE_BASIS_SELECTED


async def test_pr_falls_back_to_the_breadth_rule_when_the_key_is_absent(tmp_path):
    """A missing answer is not an empty one.

    A provider that has not seen the new prompt, or a cached response written
    before it existed, leaves the record scoped as it always was, under the
    basis that says so.
    """
    ex = _extractor(tmp_path, _payload())

    (decision,) = await ex.mine_pr_bodies()

    assert decision.affected_files == _COMMIT_FILES
    assert decision.scope_basis == SCOPE_BASIS_FOOTPRINT


async def test_pr_shows_the_model_the_commit_files(tmp_path):
    """The miner that scored worst is the one that never showed the list.

    ``mine_pr_bodies`` asked for a decision from a subject and a body alone,
    so "which files is this about" had nothing to be answered from. It scored
    20% on topic against git archaeology's 45% on the same store.
    """
    ex = _extractor(tmp_path, _payload(affected_files=[_SUBJECT]))

    await ex.mine_pr_bodies()

    (prompt,) = ex.test_provider.prompts  # type: ignore[attr-defined]
    assert "Files changed:" in prompt
    assert _SUBJECT in prompt


async def test_pr_carries_no_unvalidated_path_off_the_miner(tmp_path):
    """``proposed_files`` is cleared, so persistence cannot read it."""
    ex = _extractor(
        tmp_path, _payload(affected_files=[_SUBJECT, "packages/core/src/ghost.py"])
    )

    (decision,) = await ex.mine_pr_bodies()

    assert decision.proposed_files is None


# --- the git archaeology miner ---------------------------------------------


async def test_git_archaeology_binds_the_selected_file(tmp_path):
    ex = _extractor(tmp_path, _payload(affected_files=[_SUBJECT]))

    decisions = await ex.mine_git_archaeology()

    assert decisions, "the commit should have carried a decision signal"
    assert decisions[0].affected_files == [_SUBJECT]
    assert decisions[0].scope_basis == SCOPE_BASIS_SELECTED


async def test_git_archaeology_falls_back_when_the_key_is_absent(tmp_path):
    ex = _extractor(tmp_path, _payload())

    decisions = await ex.mine_git_archaeology()

    assert decisions[0].affected_files == _COMMIT_FILES
    assert decisions[0].scope_basis == SCOPE_BASIS_FOOTPRINT
