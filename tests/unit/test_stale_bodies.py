"""The candidate lookup and its rendering, decided from values alone.

These cover both lookups, the exclusions, every reference form this repo writes, the
truncation refusal and the table. None of them reach the network, which is what the pure
matcher is for: the fetch lives in a thin command around it and is exercised by the run
on the pull request that ships this, not here.

There is deliberately no test that a pull request never names itself. Issue and pull
request numbers come from one sequence, so a pull request's number is never an open
issue's, and such a test could not fail.
"""

from __future__ import annotations

import pytest

from tools.stale_bodies import MARKER, Candidate, StaleBodiesError, candidates, references, render

REPO = "l3a0/marketlake"


def _issue(
    number: int, *, title: str = "", body: str = "", state: str = "OPEN"
) -> dict[str, object]:
    return {"number": number, "title": title or f"issue {number}", "body": body, "state": state}


def test_cites_an_issue_the_pull_request_closes() -> None:
    """Test 1. #135 and #136 both cite #353, and #376 closed #353."""
    found = candidates(
        pull_body="Closes #353.",
        closes=[353],
        issues=[_issue(135, body="waits on #353"), _issue(999, body="unrelated")],
        repo=REPO,
    )
    assert [c.number for c in found] == [135]
    assert found[0].reasons == ("cites #353, which this closes",)


def test_named_by_the_pull_request_body() -> None:
    """Test 2. #364 said `part of #333` and closed nothing GitHub parsed."""
    found = candidates(
        pull_body="Part of #333 and part of #280.",
        closes=[],
        issues=[_issue(333), _issue(280), _issue(999)],
        repo=REPO,
    )
    assert [c.number for c in found] == [333, 280]
    assert found[0].reasons == ("this pull request's body names it",)


def test_a_closing_reference_is_never_a_candidate() -> None:
    """Test 3. The issue being closed is not a body anyone needs to re-read."""
    found = candidates(
        pull_body="Closes #353. Part of #353.",
        closes=[353],
        issues=[_issue(353, body="mentions #353 itself"), _issue(136, body="cites #353")],
        repo=REPO,
    )
    assert [c.number for c in found] == [136]


def test_a_closed_issue_is_never_named() -> None:
    """Test 4. The matcher owns this, not the query string that fed it."""
    found = candidates(
        pull_body="Part of #333.",
        closes=[353],
        issues=[
            _issue(333, state="CLOSED"),
            _issue(136, body="cites #353", state="CLOSED"),
            _issue(137, body="cites #353"),
        ],
        repo=REPO,
    )
    assert [c.number for c in found] == [137]


def test_an_issue_url_is_read_and_a_pull_url_is_not() -> None:
    """Test 5. A `pull/NNN` URL appears 221 times in this repo and names no issue."""
    body = (
        "see https://github.com/l3a0/marketlake/issues/333 and "
        "https://github.com/l3a0/marketlake/pull/345"
    )
    assert references(body, REPO) == {333}


def test_the_self_slug_is_read_and_a_foreign_one_is_not() -> None:
    """Test 6. `l3a0/marketlake#58` is real here and `actions/checkout#2454` is not."""
    body = "PR l3a0/marketlake#58 added it, and actions/checkout#2454 did not."
    assert references(body, REPO) == {58}


def test_fenced_and_inline_code_are_read() -> None:
    """Test 7. All six fenced references in this repo are real, so stripping loses them."""
    body = "```text\nshipped by #372 as #368\n```\nand `Closes #382` quoted inline\n"
    assert references(body, REPO) == {372, 368, 382}


def test_a_null_body_names_nothing() -> None:
    """Test 8. A pull request opened with no body arrives as a null."""
    assert references(None, REPO) == set()
    assert candidates(pull_body=None, closes=[], issues=[_issue(1)], repo=REPO) == []


def test_a_short_listing_is_refused_with_one_named_line() -> None:
    """Test 9. A truncated listing drops candidates in exactly the silent way this catches."""
    from tools import stale_bodies

    calls: list[list[str]] = []

    def fake_gh(args: list[str]) -> str:
        calls.append(args)
        if args[0] == "issue":
            return '[{"number": 1, "title": "t", "body": "", "state": "OPEN"}]'
        return '{"data": {"repository": {"issues": {"totalCount": 9}}}}'

    original = stale_bodies._gh
    stale_bodies._gh = fake_gh  # type: ignore[assignment]
    try:
        with pytest.raises(StaleBodiesError) as caught:
            stale_bodies._fetch_open_issues(REPO)
    finally:
        stale_bodies._gh = original  # type: ignore[assignment]
    assert "returned 1 of 9" in str(caught.value)
    assert len(str(caught.value).splitlines()) == 1
    # The listing is read before the count, so an issue closing in between lowers the
    # count and cannot cause a false refusal.
    assert [call[0] for call in calls] == ["issue", "api"]


def test_a_title_carrying_a_pipe_keeps_its_column() -> None:
    """Test 10. No title carries one today, so this is precaution rather than repair."""
    body = render(
        [Candidate(number=7, title="a | b", reasons=("named",))],
        state="merged",
    )
    assert body is not None
    row = next(line for line in body.splitlines() if line.startswith("| #7 "))
    assert row.count("|") - row.count("\\|") == 4


def test_an_empty_set_renders_no_comment() -> None:
    """Test 11. Silence means there is nothing to re-read."""
    assert render([], state="merged") is None
    assert render([], state="open") is None


def test_the_table_gives_a_reason_and_names_a_candidate_once() -> None:
    """Test 12. A candidate both lookups find appears on one row carrying both reasons."""
    found = candidates(
        pull_body="Part of #136. Closes #353.",
        closes=[353],
        issues=[_issue(136, body="cites #353", title="D18")],
        repo=REPO,
    )
    assert len(found) == 1
    assert found[0].reasons == (
        "this pull request's body names it",
        "cites #353, which this closes",
    )
    body = render(found, state="merged")
    assert body is not None
    assert body.startswith(MARKER)
    rows = [line for line in body.splitlines() if line.startswith("| #")]
    assert len(rows) == 1
    assert "names it" in rows[0] and "cites #353" in rows[0]


def test_a_pull_request_closed_unmerged_corrects_its_own_comment() -> None:
    """A list left saying a merge landed is the defect this module exists to catch."""
    body = render([], state="abandoned")
    assert body is not None
    assert MARKER in body
    assert "without merging" in body
