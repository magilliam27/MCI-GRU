"""Contract tests for the repository's own skills under .claude/skills/.

A skill whose frontmatter is malformed, or which is accidentally gated, does not
load and nothing says so — it is simply absent from the session. That silent
absence is the failure mode these tests guard.

These skills exist because five of the Matt Pocock skills this repository's
workflow depends on carry `disable-model-invocation: true`, so an agent cannot
start them. `work-the-map` and `implement-ticket` cover the parts that are safe
to reach autonomously, and must therefore not be gated themselves. See issue 168.

Two structural choices, both made after review found the first version could
report success while guarding nothing:

* The per-skill tests parametrize over `EXPECTED_SKILLS`, a **constant**, not
  over a directory listing. Globbing at collection time yields an empty
  parameter set when the directory is missing, and pytest's default for that is
  to *skip* — so deleting `.claude/skills/` turned the suite green rather than
  red.
* Paths are resolved inside the test body, so a missing file fails rather than
  disappearing from the collection.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"

EXPECTED_SKILLS = ("implement-ticket", "work-the-map")


def _skill_path(name: str) -> Path:
    return SKILLS_DIR / name / "SKILL.md"


def _read(path: Path) -> str:
    """Read a skill file, tolerating a UTF-8 BOM.

    PowerShell redirects on this machine default to UTF-8-with-BOM, and a BOM
    would otherwise make the frontmatter fence check fail with a message
    pointing at the wrong problem.
    """
    return path.read_text(encoding="utf-8-sig")


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter, body). Raises if the fence is missing or unterminated."""
    if not text.startswith("---\n"):
        raise ValueError("file does not open with a '---' frontmatter fence")
    # Search from 3, not 4: for empty frontmatter ("---\n---\n") the closing
    # fence begins at index 3, and starting at 4 misreports it as unterminated.
    end = text.find("\n---\n", 3)
    if end == -1:
        raise ValueError("frontmatter fence is never closed")
    return text[4:end], text[end + 5 :]


def _meta(path: Path) -> dict:
    """Parsed frontmatter, guaranteed to be a mapping."""
    frontmatter, body = _split_frontmatter(_read(path))
    meta = yaml.safe_load(frontmatter)
    assert isinstance(meta, dict), f"frontmatter must parse to a mapping, got {type(meta).__name__}"
    assert body.strip(), "a skill with no body carries no instructions"
    return meta


def test_the_expected_skills_are_present():
    assert SKILLS_DIR.is_dir(), f"{SKILLS_DIR} does not exist"
    for name in EXPECTED_SKILLS:
        assert _skill_path(name).is_file(), f"missing {_skill_path(name)}"


def test_every_skill_on_disk_is_well_formed():
    """Sweeps whatever is actually present, including skills added later.

    Not parametrized, deliberately: a parametrized sweep over a directory
    listing skips silently when the listing is empty, which is the case this
    most needs to catch.
    """
    found = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    assert found, f"no skills found under {SKILLS_DIR}"

    for path in found:
        # glob is case-insensitive on Windows and case-sensitive on CI's ubuntu,
        # so a file committed as `Skill.md` passes locally and vanishes there.
        assert path.name == "SKILL.md", f"{path} must be named exactly SKILL.md"
        meta = _meta(path)
        assert meta.get("name"), f"{path}: a skill without a name does not load"
        assert meta.get("description"), f"{path}: a skill without a description is never surfaced"


@pytest.mark.parametrize("name", EXPECTED_SKILLS)
def test_name_matches_its_directory(name: str):
    assert _meta(_skill_path(name))["name"] == name


@pytest.mark.parametrize("name", EXPECTED_SKILLS)
def test_these_skills_are_model_invocable(name: str):
    """These two exist precisely to be reachable without the maintainer typing them.

    Scoped to `EXPECTED_SKILLS` rather than every skill on disk: `CLAUDE.md`
    records that charting, triage, to-spec and to-tickets stay human-gated, so a
    future `.claude/skills/chart-the-map/` carrying `disable-model-invocation`
    would be correct — and a repo-wide assertion would call it a defect.

    `not ...` rather than `is not True`: PyYAML parses an unquoted `true` to
    `True`, but `"true"` to the string and `1` to the int, and both of those are
    gated in practice while passing an identity check.
    """
    assert not _meta(_skill_path(name)).get("disable-model-invocation"), (
        f"{name} is gated, defeating its purpose"
    )


def test_the_frontmatter_parser_rejects_what_it_should():
    """Control: without this, every structural test passes for a parser that never fails."""
    with pytest.raises(ValueError, match="does not open"):
        _split_frontmatter("name: x\n---\nbody\n")
    with pytest.raises(ValueError, match="never closed"):
        _split_frontmatter("---\nname: x\nbody with no closing fence\n")

    # Empty frontmatter is malformed for our purposes but IS terminated; it must
    # not be misreported as unterminated.
    frontmatter, body = _split_frontmatter("---\n---\nbody\n")
    assert frontmatter == ""
    assert body.strip() == "body"

    frontmatter, body = _split_frontmatter("---\nname: x\n---\nbody\n")
    assert yaml.safe_load(frontmatter) == {"name": "x"}
    assert body.strip() == "body"


def test_a_bom_does_not_defeat_the_frontmatter_check(tmp_path):
    """Control for `_read`: a BOM must not read as a missing fence."""
    path = tmp_path / "SKILL.md"
    path.write_text("---\nname: x\n---\nbody\n", encoding="utf-8-sig")

    assert path.read_bytes().startswith(b"\xef\xbb\xbf"), "fixture must actually carry a BOM"
    frontmatter, _ = _split_frontmatter(_read(path))
    assert yaml.safe_load(frontmatter) == {"name": "x"}


@pytest.mark.parametrize(
    "value",
    ["true", "yes", "on", "True", '"true"', "1"],
    ids=["bare", "yes", "on", "cased", "quoted", "int"],
)
def test_the_gating_check_catches_every_truthy_spelling(value: str, tmp_path):
    """Control for the model-invocable assertion.

    `is not True` passed for the quoted and integer spellings, so a gated skill
    could ship green. Each of these must read as gated.
    """
    frontmatter = f"name: x\ndescription: d\ndisable-model-invocation: {value}\n"
    meta = yaml.safe_load(frontmatter)
    assert meta.get("disable-model-invocation"), f"{value!r} must read as gated"

    # ...and the permitted spellings must not.
    for allowed in ("false", "no"):
        permitted = yaml.safe_load(f"name: x\ndisable-model-invocation: {allowed}\n")
        assert not permitted.get("disable-model-invocation")


def test_work_the_map_states_its_hard_stops():
    """The three refusals are why this skill is safe to auto-invoke.

    This is a presence check on prose and it is **weak by construction**: review
    proved it survives inverting the rules it names — "HITL tickets may be
    resolved autonomously" and "You may resolve more than one ticket" both keep
    the matched tokens. It is kept only as a tripwire against wholesale removal,
    and deliberately does NOT claim to verify the prohibitions hold.

    Do not add assertions here hoping to make it verify meaning; a substring
    search cannot. If the prohibitions move into `docs/agents/issue-tracker.md`,
    delete this test rather than extending it.
    """
    text = _read(_skill_path("work-the-map"))
    for token in ("Never chart", "HITL", "one ticket"):
        assert token.lower() in text.lower(), f"{token!r} missing from work-the-map"
