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

import json
import subprocess
import sys
import time

import pytest
from tools.pr_links import (
    AUTHOR,
    MARKER,
    Claims,
    CouldNotRun,
    Verdict,
    _gh,
    assess,
    closed_before,
    fetch,
    main,
    parsed_numbers,
    publish,
    render,
    render_could_not_run,
    scan,
)

REPO = "l3a0/marketlake"
NOW = "2026-09-17T03:18:01Z"


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


def _bot(comment: dict) -> dict:
    """A comment attributed to the check itself."""
    return {**comment, "user": {"login": AUTHOR}}


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
    gh = _Gh(comments=[_bot({"id": 1, "body": "an unrelated review comment"})])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    assert publish("body", 413, REPO, only_if_present=False) == "commented"
    assert gh.calls[-1][0] == "api"
    assert "--method" not in gh.calls[-1]


def test_a_second_run_edits_its_own_comment_rather_than_appending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gh = _Gh(comments=[_bot({"id": 7, "body": f"{MARKER}\n\n## This pull request's closing"})])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    assert publish("repaired", 413, REPO, only_if_present=False) == "updated"
    assert "PATCH" in gh.methods
    assert any("issues/comments/7" in part for part in gh.calls[-1])


def test_a_repaired_body_rewrites_a_comment_even_with_nothing_to_say(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The quiet rule covers creating a comment, never leaving a stale failure in place."""
    gh = _Gh(comments=[_bot({"id": 7, "body": f"{MARKER}\n\n## This pull request's closing"})])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    assert publish("now fine", 413, REPO, only_if_present=True) == "updated"
    assert "PATCH" in gh.methods


def test_fetch_refuses_output_that_is_not_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.pr_links._gh", lambda args: "not json at all")
    with pytest.raises(CouldNotRun, match="not JSON"):
        fetch(413, REPO)


def test_main_exits_one_on_a_lost_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr("tools.pr_links.fetch", lambda pr, repo: ("`Closes #294`", [], NOW))
    monkeypatch.setattr("tools.pr_links.closed_before", lambda n, w, r: frozenset())
    monkeypatch.setattr("tools.pr_links.publish", lambda *a, **k: "commented")
    assert main(["306"]) == 1


def test_main_exits_zero_on_a_parsed_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr("tools.pr_links.fetch", lambda pr, repo: ("Closes #390", [_ref(390)], NOW))
    monkeypatch.setattr("tools.pr_links.closed_before", lambda n, w, r: frozenset())
    monkeypatch.setattr("tools.pr_links.publish", lambda *a, **k: "commented")
    assert main(["413"]) == 0


def test_main_exits_two_when_the_query_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A red job that means "the API broke" must not be read as a lost link."""
    said: list[str] = []

    def _boom(pr: int, repo: str) -> tuple[str, list[dict], str]:
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
        missing=(2,), stale=(), contradicted=(), unclaimed=(), parsed=(1,), claimed=(1, 2)
    )


def test_a_silent_verdict_renders_as_replacing_a_stale_one() -> None:
    """A silent verdict is only ever rendered into a comment that already exists."""
    verdict = assess(scan("A docs tweak.", REPO), parsed_numbers([], REPO))
    body = render(verdict, REPO, 413)
    assert "closes nothing" in body
    assert "Nothing here is wrong" in body
    assert "did not parse" not in body


# ---------------------------------------------------------------------------
# The reference forms and line endings, from the completeness lens on PR #413.
# Each case below was silent before it was covered, which is the worst failure
# this check has: it posts "Nothing here is wrong" over a link that is broken.
# ---------------------------------------------------------------------------

WIDER_CLAIMS = [
    ("Closes\r\n#390", "a CRLF line ending, which a web-form edit submits"),
    ("Closes\r\n[#390](u)", "CRLF with the link on the next line"),
    ("Closes GH-390", "the GH-NN form"),
    ("`Closes GH-390`", "GH-NN inside a code span"),
    ("Closes https://github.com/l3a0/marketlake/issues/390", "a full issue URL"),
    ("`Closes https://github.com/l3a0/marketlake/issues/390`", "a URL in a code span"),
    ("Closes [the issue](https://github.com/l3a0/marketlake/issues/390)", "a titled link"),
    ("Closes (#390)", "parentheses"),
    ("Closes **#390**", "bold"),
    ("Closes _#390_", "italics"),
    ("Closes L3a0/Marketlake#390", "an owner/repo prefix in another case"),
]


@pytest.mark.parametrize("body,why", WIDER_CLAIMS, ids=[w for _, w in WIDER_CLAIMS])
def test_scan_reads_the_wider_reference_forms(body: str, why: str) -> None:
    assert scan(body, REPO).closes == frozenset({390}), why


def test_a_foreign_issue_url_is_not_a_local_claim() -> None:
    body = "Fixes https://github.com/actions/checkout/issues/2454"
    assert scan(body, REPO).closes == frozenset()


def test_a_crlf_body_still_refuses_a_blank_line() -> None:
    """Widening to CRLF must not widen to a blank line, which is the #194 shape."""
    assert scan("## How #184 was resolved\r\n\r\n[#192](u)", REPO).closes == frozenset()


# ---------------------------------------------------------------------------
# Rule 2: a claim on an issue that closed before the pull request existed.
# ---------------------------------------------------------------------------


def test_a_claim_on_an_already_closed_issue_reports_and_does_not_fail() -> None:
    """#112 narrates #85's body fifteen hours after #77 had already closed."""
    body = "**#77**, closed by [#85](u), whose body opens `Closes #77.` and then says more."
    verdict = assess(scan(body, REPO), parsed_numbers([], REPO), frozenset({77}))
    assert verdict.stale == (77,)
    assert verdict.missing == ()
    assert verdict.broken is False


def test_a_claim_on_a_live_issue_still_fails() -> None:
    verdict = assess(scan("`Closes #294`", REPO), parsed_numbers([], REPO), frozenset())
    assert verdict.missing == (294,)
    assert verdict.stale == ()
    assert verdict.broken is True


def test_a_stale_claim_is_named_in_the_comment() -> None:
    body = render(
        assess(scan("`Closes #77`", REPO), parsed_numbers([], REPO), frozenset({77})), REPO, 112
    )
    assert "#77" in body
    assert "already closed before this pull request existed" in body
    assert "## This pull request's closing link did not parse" not in body


def test_closed_before_asks_only_about_unparsed_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("tools.pr_links._gh", lambda args: calls.append(args) or "{}")
    assert closed_before((), NOW, REPO) == frozenset()
    assert calls == []


def test_closed_before_reads_the_closed_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": {
            "repository": {
                "i77": {"number": 77, "closedAt": "2026-09-12T04:29:41Z"},
                "i294": {"number": 294, "closedAt": None},
                "i999": {"number": 999, "closedAt": "2099-01-01T00:00:00Z"},
            }
        }
    }
    monkeypatch.setattr("tools.pr_links._gh", lambda args: json.dumps(payload))
    assert closed_before((77, 294, 999), NOW, REPO) == frozenset({77})


def test_closed_before_fails_open_so_a_broken_query_still_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning nothing leaves every claim in `missing`, which is the loud direction."""

    def _boom(args: list[str]) -> str:
        raise CouldNotRun("gh exited 1")

    monkeypatch.setattr("tools.pr_links._gh", _boom)
    assert closed_before((77,), NOW, REPO) == frozenset()


# ---------------------------------------------------------------------------
# The comment this check will edit has to be one it wrote.
# ---------------------------------------------------------------------------


def test_a_human_comment_carrying_the_marker_is_never_overwritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A person quoting the check's raw output carries the marker too, and is listed first."""
    gh = _Gh(
        comments=[
            {
                "id": 111,
                "body": f"quoting it:\n\n{MARKER}\n\n## verdict",
                "user": {"login": "l3a0"},
            },
            _bot({"id": 222, "body": f"{MARKER}\n\n## the real verdict"}),
        ]
    )
    monkeypatch.setattr("tools.pr_links._gh", gh)
    assert publish("new verdict", 413, REPO, only_if_present=False) == "updated"
    assert any("issues/comments/222" in part for part in gh.calls[-1])
    assert not any("issues/comments/111" in part for call in gh.calls for part in call)


def test_no_comment_of_its_own_means_it_creates_one(monkeypatch: pytest.MonkeyPatch) -> None:
    gh = _Gh(comments=[{"id": 111, "body": MARKER, "user": {"login": "l3a0"}}])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    assert publish("verdict", 413, REPO, only_if_present=False) == "commented"
    assert "--method" not in gh.calls[-1]


# ---------------------------------------------------------------------------
# The API seam itself. Every other test stubs `_gh`, so without these the
# function that talks to `gh` has nothing behind it at all.
# ---------------------------------------------------------------------------


def test_gh_returns_stdout_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def _run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
        seen.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, 0, stdout="OUT", stderr="ERR")

    monkeypatch.setattr(subprocess, "run", _run)
    assert _gh(["pr", "view"]) == "OUT"
    assert seen["cmd"] == ["gh", "pr", "view"]
    assert seen["timeout"] == 60
    assert seen["check"] is False


def test_gh_refuses_a_nonzero_exit_and_carries_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="rate limited"),
    )
    with pytest.raises(CouldNotRun, match="gh exited 1: rate limited"):
        _gh(["pr", "view"])


def test_gh_refuses_when_it_cannot_be_run_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """With `gh` missing, silence would make every pull request pass while doing nothing."""

    def _missing(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
        raise FileNotFoundError("gh")

    monkeypatch.setattr(subprocess, "run", _missing)
    with pytest.raises(CouldNotRun, match="could not be run"):
        _gh(["pr", "view"])


def test_gh_refuses_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _hang(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(subprocess, "run", _hang)
    with pytest.raises(CouldNotRun):
        _gh(["pr", "view"])


def test_fetch_returns_the_body_the_references_and_the_creation_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    payload = {
        "body": "Closes #390",
        "closingIssuesReferences": [_ref(390)],
        "createdAt": NOW,
    }
    monkeypatch.setattr(
        "tools.pr_links._gh", lambda args: calls.append(args) or json.dumps(payload)
    )
    body, references, created = fetch(413, REPO)
    assert body == "Closes #390"
    assert [r["number"] for r in references] == [390]
    assert created == NOW
    assert "--repo" in calls[0]
    assert "body,closingIssuesReferences,createdAt" in calls[0]


def test_the_comment_listing_is_paginated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --paginate the marker is missed past one page and a second comment appears."""
    gh = _Gh(comments=[])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    publish("verdict", 413, REPO, only_if_present=False)
    assert "--paginate" in gh.calls[0]


def test_the_comment_body_reaches_gh_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    gh = _Gh(comments=[])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    publish("a body\nwith newlines and `backticks` and #390", 413, REPO, only_if_present=False)
    assert "body=a body\nwith newlines and `backticks` and #390" in gh.calls[-1]


def test_a_patch_carries_the_body_too(monkeypatch: pytest.MonkeyPatch) -> None:
    gh = _Gh(comments=[_bot({"id": 7, "body": MARKER})])
    monkeypatch.setattr("tools.pr_links._gh", gh)
    publish("the new verdict", 413, REPO, only_if_present=False)
    assert "body=the new verdict" in gh.calls[-1]


# ---------------------------------------------------------------------------
# What the comment says. Every branch below could be deleted without a test
# noticing before these existed.
# ---------------------------------------------------------------------------


def test_a_half_parsed_body_names_only_the_half_that_failed() -> None:
    """#377's shape. `Missing:` must not name the reference that parsed."""
    body = "`Closes #281` and `Closes #308`."
    rendered = render(assess(scan(body, REPO), parsed_numbers([_ref(308)], REPO)), REPO, 377)
    assert "Missing: **#281**" in rendered
    assert "claims to close #281, #308" in rendered
    assert "Missing: **#281, #308**" not in rendered


def test_a_multi_reference_verdict_names_every_one() -> None:
    rendered = render(assess(scan("Closes #281 and closes #308", REPO), frozenset()), REPO, 377)
    assert "#281, #308" in rendered


def test_the_contradicted_report_reaches_the_comment() -> None:
    verdict = assess(
        scan("Closes #128. The trailer said `Part of #128`.", REPO),
        parsed_numbers([_ref(128)], REPO),
    )
    assert "Part of" in render(verdict, REPO, 147)


def test_the_unclaimed_report_reaches_the_comment() -> None:
    """The only signal that the check itself is narrower than GitHub."""
    verdict = assess(scan("no keyword", REPO), parsed_numbers([_ref(390)], REPO))
    assert "narrower than GitHub" in render(verdict, REPO, 413)


# ---------------------------------------------------------------------------
# `main` wiring: the quiet rule is decided here, not in `publish`.
# ---------------------------------------------------------------------------


class _Publish:
    def __init__(self) -> None:
        self.only_if_present: bool | None = None
        self.body = ""

    def __call__(self, body: str, pr: int, repo: str, *, only_if_present: bool) -> str:
        self.only_if_present = only_if_present
        self.body = body
        return "commented"


def test_main_lets_a_broken_link_create_a_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _Publish()
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr("tools.pr_links.fetch", lambda pr, repo: ("`Closes #294`", [], NOW))
    monkeypatch.setattr("tools.pr_links.closed_before", lambda n, w, r: frozenset())
    monkeypatch.setattr("tools.pr_links.publish", spy)
    assert main(["306"]) == 1
    assert spy.only_if_present is False


def test_main_keeps_a_silent_pull_request_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Dependabot bump reaches here, and its token cannot write a comment."""
    spy = _Publish()
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr("tools.pr_links.fetch", lambda pr, repo: ("Bumps a dependency.", [], NOW))
    monkeypatch.setattr("tools.pr_links.publish", spy)
    assert main(["31"]) == 0
    assert spy.only_if_present is True


def test_main_asks_about_staleness_only_for_unparsed_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[tuple] = []
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr(
        "tools.pr_links.fetch", lambda pr, repo: ("Closes #390 and closes #77", [_ref(390)], NOW)
    )
    monkeypatch.setattr(
        "tools.pr_links.closed_before",
        lambda numbers, when, repo: asked.append((numbers, when)) or frozenset({77}),
    )
    monkeypatch.setattr("tools.pr_links.publish", _Publish())
    assert main(["112"]) == 0
    assert asked == [((77,), NOW)]


def test_main_annotates_the_failure_for_the_actions_log(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setattr("tools.pr_links.fetch", lambda pr, repo: ("`Closes #294`", [], NOW))
    monkeypatch.setattr("tools.pr_links.closed_before", lambda n, w, r: frozenset())
    monkeypatch.setattr("tools.pr_links.publish", _Publish())
    assert main(["306"]) == 1
    assert "::error::closing keyword did not parse for #294" in capsys.readouterr().out


def test_main_refuses_without_a_pull_request_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    assert main([]) == 2


def test_parsed_numbers_matches_a_repository_in_another_case() -> None:
    """GitHub treats an owner and a repository name as case-insensitive."""
    assert parsed_numbers([_ref(390, "L3a0/Marketlake")], REPO) == frozenset({390})


# ---------------------------------------------------------------------------
# The scan must stay linear. Before the quantifiers were made possessive it was
# cubic in a whitespace run, so a keyword followed by spaces and no reference
# hung the job past its five-minute ceiling: a red check, no comment, and the
# previous run's verdict left standing beside it. Anyone who can open a pull
# request on a public repository can write that body and re-trigger it.
# ---------------------------------------------------------------------------

BLOWUP = [
    ("Closes" + " " * 400 + "and then prose.", "a run of spaces reaching no reference"),
    ("Closes" + "\t" * 400 + "and then prose.", "tabs"),
    ("Closes" + "[" * 400 + " none", "a run of brackets"),
    ("Closes" + "`(*_[" * 200 + " none", "mixed decoration"),
    ("Closes " + ":" * 400 + " none", "colons"),
    ("Part of" + " " * 400 + "prose.", "the same shape on the Part of pattern"),
]


@pytest.mark.parametrize("body,why", BLOWUP, ids=[w for _, w in BLOWUP])
def test_the_scan_stays_linear(body: str, why: str) -> None:
    """Four hundred characters, not four thousand, and that size is the point.

    The cubic version took 0.856s on 400 characters and 22.7s on 1,200, so this size fails
    fast under a regression while a linear scan finishes in tens of microseconds. Feeding it
    a realistic 20,000 does not make the test stronger, it makes it **hang** rather than
    fail, and a hanging test burns the job's timeout instead of reporting anything. The
    production-scale guarantee is held by the subprocess test below, which cannot hang
    because it is killed.
    """
    started = time.perf_counter()
    assert scan(body, REPO).closes == frozenset()
    assert time.perf_counter() - started < 0.25, why


def test_the_scan_survives_a_body_at_github_s_size_limit() -> None:
    """A full-size hostile body, run where a hang is a failure rather than a wait.

    `re` does not check for signals while matching, so a catastrophic backtrack cannot be
    interrupted in process. A subprocess can be killed, which is what makes this assertion
    possible at all.
    """
    program = (
        "from tools.pr_links import scan;scan('Closes' + ' ' * 65000 + 'prose', 'l3a0/marketlake')"
    )
    subprocess.run([sys.executable, "-c", program], timeout=10, check=True, capture_output=True)


def test_a_qualified_reference_inside_a_markdown_link_still_reads() -> None:
    """The link branch must not swallow a label that carries the reference itself."""
    assert scan("Closes [l3a0/marketlake#390](u)", REPO).closes == frozenset({390})


# ---------------------------------------------------------------------------
# Payloads shaped unlike GitHub's schema must refuse rather than traceback. An
# exception escaping `publish` skips the `broken` check in `main` entirely, so a
# pull request whose link parsed correctly would still go red.
# ---------------------------------------------------------------------------


def test_a_comment_listing_that_is_not_a_list_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.pr_links._gh", lambda args: json.dumps({"message": "Not Found"}))
    with pytest.raises(CouldNotRun, match="not shaped like comments"):
        publish("verdict", 413, REPO, only_if_present=False)


def test_a_comment_with_no_id_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [{"body": MARKER, "user": {"login": AUTHOR}}]
    monkeypatch.setattr("tools.pr_links._gh", lambda args: json.dumps(listing))
    with pytest.raises(CouldNotRun, match="not shaped like comments"):
        publish("verdict", 413, REPO, only_if_present=False)


def test_parsed_numbers_survives_null_repository_fields() -> None:
    refs = [
        {"number": 1, "repository": {"name": None, "owner": {"login": None}}},
        {"repository": {"name": "marketlake", "owner": {"login": "l3a0"}}},
        _ref(390),
    ]
    assert parsed_numbers(refs, REPO) == frozenset({390})


def test_closed_before_survives_a_payload_it_does_not_recognise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.pr_links._gh", lambda args: json.dumps({"errors": ["nope"]}))
    assert closed_before((77,), NOW, REPO) == frozenset()


# ---------------------------------------------------------------------------
# `main`'s own arguments.
# ---------------------------------------------------------------------------


def test_main_refuses_an_argument_that_is_not_a_number(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The workflow passes "$PR_NUMBER" quoted, so an empty value arrives as an empty string."""
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    assert main([""]) == 2
    assert "not a pull request number" in capsys.readouterr().err


def test_main_refuses_a_repository_without_a_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare name makes every qualified reference foreign and fails a correct body."""
    monkeypatch.setenv("GITHUB_REPOSITORY", "marketlake")
    assert main(["413"]) == 2
