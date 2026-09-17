"""The closing-link check: what a body claims set against what GitHub parsed.

Every case below is a string, and no test reaches the network. The API is stubbed at
``tools.pr_links._gh``, which is the single seam every ``gh`` call goes through.

Three groups of cases come from real pull requests rather than invention.

1. **The spellings that lost a link.** #306 wrote the keyword inside a code span, #364 split
   it from a markdown link across a line break, and #372 split it inside one code span. The
   scan has to see all three, because GitHub did not.
2. **The spellings that must not be read as a claim.** #194's ``## How #184 was resolved``
   heading is followed by a blank line and then a link, and reading that as a claim would
   fail a correct pull request. The Dependabot bump #31 carries fifteen upstream numbers.
3. **This repository's own vocabulary.** Its bodies and `CLAUDE.md` talk about a
   "Conventional Commits prefix" constantly, so a keyword pattern that matches inside
   ``prefix`` would fail nearly everything.
"""

from __future__ import annotations

import pytest
from tools.pr_links import (
    MARKER,
    Claims,
    CouldNotRun,
    Verdict,
    assess,
    fetch,
    main,
    parsed_numbers,
    publish,
    render,
    render_could_not_run,
    scan,
)

REPO = "l3a0/marketlake"


def _ref(number: int, repo: str = REPO) -> dict:
    owner, _, name = repo.partition("/")
    return {"number": number, "repository": {"name": name, "owner": {"login": owner}}}


# The spellings a body may use, and whether the scan must read each as a local claim.
CLAIMED = [
    ("Closes #390", "plain text, the spelling that works"),
    ("closes #390", "lowercase"),
    ("CLOSES #390", "uppercase"),
    ("Closes: #390", "a colon after the keyword"),
    ("`Closes #390`", "inside a code span, the #306 shape"),
    ("Closes [#390](https://github.com/l3a0/marketlake/issues/390)", "a markdown link"),
    ("Closes\n#390", "the number on the next line"),
    ("Closes\n[#390](u)", "a link on the next line, the #364 shape"),
    ("`Closes\n#390`", "split inside one code span, the #372 shape"),
    ("Fixes #390", "fixes"),
    ("Fixed #390", "fixed"),
    ("Resolves #390", "resolves"),
    ("Resolved #390", "resolved"),
    ("Close #390", "the bare stem"),
    ("Closes l3a0/marketlake#390", "an owner/repo prefix naming this repository"),
]

NOT_CLAIMED = [
    ("Conventional Commits prefix. See #390", "prefix must not match fix"),
    ("suffix #390", "suffix must not match fix"),
    ("unfixed #390", "unfixed must not match fixed"),
    ("closest #390", "closest must not match close"),
    ("disclosed #390", "disclosed must not match closed"),
    ("enclosure #390", "enclosure must not match close"),
    ("## How #184 was resolved\n\n[#192](u)", "a blank line, the #194 shape"),
    ("resolved\n\n#390", "a blank line is never crossed"),
    ("Fixes actions/checkout#2454", "a foreign repository is not a local claim"),
    ("See #390 for the reasoning", "a reference with no keyword"),
    ("#390", "a bare number"),
]


@pytest.mark.parametrize("body,why", CLAIMED, ids=[w for _, w in CLAIMED])
def test_scan_reads_a_claim(body: str, why: str) -> None:
    assert scan(body, REPO).closes == frozenset({390}), why


@pytest.mark.parametrize("body,why", NOT_CLAIMED, ids=[w for _, w in NOT_CLAIMED])
def test_scan_refuses_a_non_claim(body: str, why: str) -> None:
    assert scan(body, REPO).closes == frozenset(), why


def test_scan_reads_part_of_separately() -> None:
    claims = scan("Part of #136. Closes #382.", REPO)
    assert claims.closes == frozenset({382})
    assert claims.part_of == frozenset({136})


def test_scan_reads_part_of_across_a_line_break_and_a_link() -> None:
    assert scan("Part of\n[#136](u)", REPO).part_of == frozenset({136})


def test_scan_handles_an_empty_body() -> None:
    assert scan(None, REPO) == Claims()
    assert scan("", REPO) == Claims()


def test_scan_reads_every_claim_in_one_body() -> None:
    body = "`Closes #281` and `Closes #308`. Nothing is left in either."
    assert scan(body, REPO).closes == frozenset({281, 308})


def test_dependabot_upstream_numbers_are_not_claims() -> None:
    """#31's body carries fifteen upstream numbers and no keyword before any of them."""
    body = "Bumps actions/checkout.\n\n* Add support (#2474)\n* See #598 and #601 for detail."
    assert scan(body, REPO).closes == frozenset()


def test_parsed_numbers_keeps_this_repository() -> None:
    assert parsed_numbers([_ref(319), _ref(390)], REPO) == frozenset({319, 390})


def test_parsed_numbers_drops_another_repository() -> None:
    """A foreign issue must not collide with a local one sharing its number."""
    refs = [_ref(2454, "actions/checkout"), _ref(390)]
    assert parsed_numbers(refs, REPO) == frozenset({390})


def test_parsed_numbers_survives_a_missing_repository_block() -> None:
    assert parsed_numbers([{"number": 1}], REPO) == frozenset()


def test_a_claim_github_did_not_parse_is_the_failure() -> None:
    verdict = assess(scan("`Closes #294`", REPO), parsed_numbers([], REPO))
    assert verdict.missing == (294,)
    assert verdict.broken is True


def test_a_claim_github_parsed_passes() -> None:
    verdict = assess(scan("Closes #390", REPO), parsed_numbers([_ref(390)], REPO))
    assert verdict.missing == ()
    assert verdict.broken is False
    assert verdict.parsed == (390,)


def test_a_half_parsed_body_fails_on_the_half_that_did_not() -> None:
    """#377 wrote two keywords the same way and GitHub parsed one of them."""
    body = "`Closes #281` and `Closes #308`. Nothing is left in either."
    verdict = assess(scan(body, REPO), parsed_numbers([_ref(308)], REPO))
    assert verdict.claimed == (281, 308)
    assert verdict.missing == (281,)
    assert verdict.broken is True


def test_part_of_parsed_as_closing_reports_and_does_not_fail() -> None:
    """Rule 2. Zero true hits over 182, and its one firing is a quoted trailer in #147."""
    verdict = assess(
        scan("Closes #128. The trailer said `Part of #128`.", REPO),
        parsed_numbers([_ref(128)], REPO),
    )
    assert verdict.contradicted == (128,)
    assert verdict.broken is False


def test_something_parsed_the_scan_did_not_find_reports_and_does_not_fail() -> None:
    verdict = assess(scan("no keyword here", REPO), parsed_numbers([_ref(390)], REPO))
    assert verdict.unclaimed == (390,)
    assert verdict.broken is False


def test_claiming_nothing_and_parsing_nothing_is_silent() -> None:
    """Rule 3. Eighty-one of 182 pull requests link no issue at all."""
    verdict = assess(scan("A docs tweak.", REPO), parsed_numbers([], REPO))
    assert verdict.silent is True
    assert verdict.broken is False


def test_a_part_of_only_body_is_not_silent() -> None:
    """`Part of #NN` is the house spelling for a partial fix, so it has something to say."""
    verdict = assess(scan("Part of #136.", REPO), parsed_numbers([], REPO))
    assert verdict.silent is False
    assert verdict.broken is False


def test_a_failure_comment_names_the_missing_numbers_and_the_command() -> None:
    verdict = assess(scan("`Closes #294`", REPO), parsed_numbers([], REPO))
    body = render(verdict, REPO, 306)
    assert body.startswith(MARKER)
    assert "#294" in body
    assert f"gh pr view 306 --repo {REPO} --json closingIssuesReferences" in body
    assert "plain text" in body


def test_a_passing_comment_names_what_github_parsed() -> None:
    """The confirmation is the point. Sixteen pull requests never got it."""
    verdict = assess(scan("Closes #390", REPO), parsed_numbers([_ref(390)], REPO))
    body = render(verdict, REPO, 413)
    assert "parsed" in body
    assert "#390" in body
    assert "did not parse" not in body


def test_a_could_not_run_comment_says_so_rather_than_blaming_the_link() -> None:
    body = render_could_not_run("gh exited 1: rate limited", 413, REPO)
    assert MARKER in body
    assert "could not run" in body
    assert "rate limited" in body
    assert "says nothing about whether the closing link parsed" in body


class _Gh:
    """Records every ``gh`` call and answers from a script."""

    def __init__(self, comments: list[dict] | None = None) -> None:
        self.comments = comments if comments is not None else []
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        if args[0] == "api" and args[1].endswith("/comments") and "--method" not in args:
            import json

            return json.dumps(self.comments)
        return "{}"

    @property
    def methods(self) -> list[str]:
        return ["PATCH" if "--method" in c else c[0] for c in self.calls]


def test_a_silent_pull_request_gets_no_new_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    gh = _Gh(comments=[])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    assert publish("body", 413, REPO, only_if_present=True) == "stayed quiet"
    assert all("--method" not in call for call in gh.calls)
    assert not any(call[0] == "api" and "-f" in call for call in gh.calls)


def test_a_verdict_creates_a_comment_when_none_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    gh = _Gh(comments=[{"id": 1, "body": "an unrelated review comment"}])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    assert publish("body", 413, REPO, only_if_present=False) == "commented"
    assert gh.calls[-1][0] == "api"
    assert "--method" not in gh.calls[-1]


def test_a_second_run_edits_its_own_comment_rather_than_appending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gh = _Gh(comments=[{"id": 7, "body": f"{MARKER}\n\n## This pull request's closing link"}])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    assert publish("repaired", 413, REPO, only_if_present=False) == "updated"
    assert "PATCH" in gh.methods
    assert any("issues/comments/7" in part for part in gh.calls[-1])


def test_a_repaired_body_rewrites_a_comment_even_with_nothing_to_say(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The quiet rule covers creating a comment, never leaving a stale failure in place."""
    gh = _Gh(comments=[{"id": 7, "body": f"{MARKER}\n\n## This pull request's closing link"}])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    assert publish("now fine", 413, REPO, only_if_present=True) == "updated"
    assert "PATCH" in gh.methods


def test_fetch_refuses_output_that_is_not_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.pr_links._gh", lambda args: "not json at all")
    with pytest.raises(CouldNotRun, match="not JSON"):
        fetch(413, REPO)


def test_main_exits_one_on_a_lost_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr("tools.pr_links.fetch", lambda pr, repo: ("`Closes #294`", []))
    monkeypatch.setattr("tools.pr_links.publish", lambda *a, **k: "commented")
    assert main(["306"]) == 1


def test_main_exits_zero_on_a_parsed_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr("tools.pr_links.fetch", lambda pr, repo: ("Closes #390", [_ref(390)]))
    monkeypatch.setattr("tools.pr_links.publish", lambda *a, **k: "commented")
    assert main(["413"]) == 0


def test_main_exits_two_when_the_query_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A red job that means "the API broke" must not be read as a lost link."""
    said: list[str] = []

    def _boom(pr: int, repo: str) -> tuple[str, list[dict]]:
        raise CouldNotRun("gh exited 1: rate limited")

    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr("tools.pr_links.fetch", _boom)
    monkeypatch.setattr(
        "tools.pr_links.publish", lambda body, *a, **k: said.append(body) or "commented"
    )
    assert main(["413"]) == 2
    assert "could not run" in said[0]


def test_main_refuses_without_a_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert main(["413"]) == 2


def test_the_verdict_carries_what_it_compared() -> None:
    verdict = assess(scan("Closes #1 and closes #2", REPO), parsed_numbers([_ref(1)], REPO))
    assert verdict == Verdict(
        missing=(2,), contradicted=(), unclaimed=(), parsed=(1,), claimed=(1, 2)
    )


def test_a_silent_verdict_renders_as_replacing_a_stale_one() -> None:
    """A silent verdict is only ever rendered into a comment that already exists."""
    verdict = assess(scan("A docs tweak.", REPO), parsed_numbers([], REPO))
    body = render(verdict, REPO, 413)
    assert "closes nothing" in body
    assert "Nothing here is wrong" in body
    assert "did not parse" not in body
