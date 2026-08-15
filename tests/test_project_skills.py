"""Contract tests for the repository's own skills under .claude/skills/.

A skill whose frontmatter is malformed does not load, and nothing says so — the
skill is simply absent from the session. That is a real failure mode, not a
wording check, and it is what these tests guard.

These skills exist because five of the Matt Pocock skills this repository's
workflow depends on carry `disable-model-invocation: true`: an agent cannot
start them. `work-the-map` and `implement-ticket` cover the parts that are safe
to reach autonomously. They must therefore NOT carry that flag themselves — a
model-invoked skill that is accidentally gated is invisible in exactly the same
way as a malformed one. See issue 168.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"

EXPECTED_SKILLS = {"work-the-map", "implement-ticket"}


def _skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*/SKILL.md"))


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter, body). Raises if the fence is missing or unterminated."""
    if not text.startswith("---\n"):
        raise ValueError("file does not open with a '---' frontmatter fence")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("frontmatter fence is never closed")
    return text[4:end], text[end + 5 :]


def test_the_expected_skills_are_present():
    assert SKILLS_DIR.is_dir(), f"{SKILLS_DIR} does not exist"
    found = {p.parent.name for p in _skill_files()}
    assert found >= EXPECTED_SKILLS, f"missing: {EXPECTED_SKILLS - found}"


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.parent.name)
def test_frontmatter_parses_and_declares_name_and_description(skill_path: Path):
    frontmatter, body = _split_frontmatter(skill_path.read_text(encoding="utf-8"))
    meta = yaml.safe_load(frontmatter)

    assert isinstance(meta, dict), "frontmatter must parse to a mapping"
    assert meta.get("name"), "a skill without a name does not load"
    assert meta.get("description"), "a skill without a description is never surfaced"
    assert body.strip(), "a skill with no body carries no instructions"


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.parent.name)
def test_name_matches_its_directory(skill_path: Path):
    frontmatter, _ = _split_frontmatter(skill_path.read_text(encoding="utf-8"))
    assert yaml.safe_load(frontmatter)["name"] == skill_path.parent.name


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.parent.name)
def test_project_skills_are_model_invocable(skill_path: Path):
    """These exist precisely to be reachable without the maintainer typing them.

    A stray `disable-model-invocation: true` would make them invisible to an
    agent while looking completely fine on disk -- the same silent absence the
    frontmatter tests guard against.
    """
    frontmatter, _ = _split_frontmatter(skill_path.read_text(encoding="utf-8"))
    meta = yaml.safe_load(frontmatter)
    assert meta.get("disable-model-invocation") is not True, (
        f"{skill_path.parent.name} is gated, defeating its purpose"
    )


def test_the_frontmatter_parser_rejects_what_it_should():
    """Control: without this, every test above passes for a parser that never fails."""
    with pytest.raises(ValueError, match="does not open"):
        _split_frontmatter("name: x\n---\nbody\n")
    with pytest.raises(ValueError, match="never closed"):
        _split_frontmatter("---\nname: x\nbody with no closing fence\n")

    frontmatter, body = _split_frontmatter("---\nname: x\n---\nbody\n")
    assert yaml.safe_load(frontmatter) == {"name": "x"}
    assert body.strip() == "body"


def test_work_the_map_refuses_charting_and_hitl():
    """The three hard stops are the reason this skill is safe to make reachable.

    This is a wording assertion, which is normally the vacuous-guard pattern this
    repository has been bitten by. It is justified here for a specific, checked
    reason: a line-by-line comparison against `docs/agents/issue-tracker.md` and
    `CLAUDE.md` found that "never chart" exists in **neither** — and that the
    tracker doc actively teaches charting mechanics with no counterweight. So
    this skill is the unique home of that prohibition, and the assertion defends
    a real invariant rather than pinning phrasing.

    If the prohibition is ever moved into the tracker doc, delete this test
    rather than loosening it -- at that point it would be pinning phrasing.
    """
    text = (SKILLS_DIR / "work-the-map" / "SKILL.md").read_text(encoding="utf-8")
    assert "Never chart" in text
    assert "HITL" in text
    assert "one ticket" in text.lower()
