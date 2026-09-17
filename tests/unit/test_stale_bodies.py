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

import json

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


def test_a_hash_after_a_hyphen_is_still_a_reference() -> None:
    """Test 21. #194's body writes "The pre-#184 behaviour stays", and #184 is real.

    A lookbehind class of ``[\\w/-]`` looked free. Across all 425 bodies here, no hash
    follows a slash and the only one following a hyphen is that reference, so those two
    characters cost a reference and excluded nothing ``\\w`` did not already exclude.
    """
    assert references("The pre-#184 behaviour stays.", REPO) == {184}
    assert references("compare #392-#393 side by side", REPO) == {392, 393}
    assert references("actions/checkout#2454", REPO) == set()


def test_a_closing_reference_in_another_repository_is_discarded() -> None:
    """Test 22. Its bare number would read as a local issue and exclude the wrong one."""
    from tools import stale_bodies

    payload = {
        "closingIssuesReferences": [
            {"number": 136, "repository": {"name": "marketlake", "owner": {"login": "l3a0"}}},
            {"number": 99, "repository": {"name": "other", "owner": {"login": "l3a0"}}},
            {"number": 7, "repository": {"name": "marketlake", "owner": {"login": "someone"}}},
            {"number": 5},
        ]
    }
    original = stale_bodies._gh
    stale_bodies._gh = lambda args: json.dumps(payload)  # type: ignore[assignment]
    try:
        assert stale_bodies._fetch_closes(REPO, 1) == [136]
    finally:
        stale_bodies._gh = original  # type: ignore[assignment]


def test_a_missing_gh_is_refused_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 23. Every refusal is one named line, including the one before gh runs."""
    from tools import stale_bodies

    def no_gh(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError(2, "No such file or directory", "gh")

    monkeypatch.setattr(stale_bodies.subprocess, "run", no_gh)
    with pytest.raises(StaleBodiesError) as caught:
        stale_bodies._gh(["issue", "list"])
    assert str(caught.value).startswith("gh could not be run")
    assert len(str(caught.value).splitlines()) == 1


def test_fenced_and_inline_code_are_read() -> None:
    """Test 7. All six fenced references in this repo are real, so stripping loses them."""
    body = "```text\nshipped by #372 as #368\n```\nand `Closes #382` quoted inline\n"
    assert references(body, REPO) == {372, 368, 382}


def test_a_null_body_names_nothing() -> None:
    """Test 8. A pull request opened with no body arrives as a null."""
    assert references(None, REPO) == set()
    assert candidates(pull_body=None, closes=[], issues=[_issue(1)], repo=REPO) == []


def _fake_fetch(
    monkeypatch: pytest.MonkeyPatch, listings: list[str], counts: list[int]
) -> list[str]:
    """Drive ``_fetch_open_issues`` through a scripted pair of listings and counts."""
    from tools import stale_bodies

    seen: list[str] = []

    def fake_gh(args: list[str]) -> str:
        seen.append(args[0])
        if args[0] == "issue":
            return listings.pop(0)
        return json.dumps({"data": {"repository": {"issues": {"totalCount": counts.pop(0)}}}})

    monkeypatch.setattr(stale_bodies, "_gh", fake_gh)
    return seen


def _rows(count: int) -> str:
    return json.dumps(
        [{"number": n, "title": "t", "body": "", "state": "OPEN"} for n in range(count)]
    )


def test_a_listing_short_twice_is_refused_with_one_named_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 9. A truncated listing drops candidates in exactly the silent way this catches."""
    from tools import stale_bodies

    seen = _fake_fetch(monkeypatch, [_rows(1), _rows(1)], [9, 9])
    with pytest.raises(StaleBodiesError) as caught:
        stale_bodies._fetch_open_issues(REPO)
    assert "returned 1 of 9 twice" in str(caught.value)
    assert len(str(caught.value).splitlines()) == 1
    assert seen == ["issue", "api", "issue", "api"]


def test_a_listing_short_once_is_read_again_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 13. An issue filed mid-run is a race, not a truncation.

    This module's own first run refused with "returned 111 of 112" because another
    session filed an issue between the listing and the count. A guard that refuses
    whenever somebody files an issue is a guard whose reader turns it off.
    """
    from tools import stale_bodies

    seen = _fake_fetch(monkeypatch, [_rows(111), _rows(112)], [112, 112])
    issues = stale_bodies._fetch_open_issues(REPO)
    assert len(issues) == 112
    assert seen == ["issue", "api", "issue", "api"]


def test_dry_run_prints_the_comment_instead_of_writing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test 14. The mode that lets a person run the check by hand against a real merge."""
    from tools import stale_bodies

    monkeypatch.setattr(stale_bodies, "_fetch_closes", lambda repo, n: [353])
    monkeypatch.setattr(
        stale_bodies,
        "_fetch_open_issues",
        lambda repo: [_issue(136, title="D18", body="cites #353")],
    )
    written: list[object] = []
    monkeypatch.setattr(stale_bodies, "_upsert", lambda *a, **k: written.append(a))

    assert stale_bodies.main(["--pull", "1", "--repo", REPO, "--state", "merged", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert MARKER in out
    assert "| #136 |" in out
    assert written == []


def test_a_generator_of_issues_does_not_blank_every_title() -> None:
    """Test 18. The rows are read twice, for the reasons and for the titles."""
    rows = [_issue(136, title="D18", body="cites #353")]
    found = candidates(pull_body="", closes=[353], issues=(r for r in rows), repo=REPO)
    assert [c.title for c in found] == ["D18"]


def test_every_refusal_reaches_the_operator_as_one_named_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test 19. The issue asks for one named line rather than a traceback, on every path."""
    from tools import stale_bodies

    # A body file that is not there, refused before anything reaches the network.
    assert stale_bodies.main(["--pull", "1", "--repo", "a/b", "--body-file", "/no/such"]) == 1
    err = capsys.readouterr().err.strip()
    assert err.startswith("stale_bodies: the pull request body file could not be read")
    assert len(err.splitlines()) == 1

    # A reply that is not JSON.
    monkeypatch.setattr(stale_bodies, "_gh", lambda args: "<html>rate limited</html>")
    with pytest.raises(StaleBodiesError) as caught:
        stale_bodies._fetch_closes(REPO, 1)
    assert str(caught.value).startswith("the closing references did not come back as JSON")

    # A reply whose shape is wrong.
    monkeypatch.setattr(stale_bodies, "_gh", lambda args: '{"data": {}}')
    with pytest.raises(StaleBodiesError) as caught:
        stale_bodies._count_open_issues(REPO)
    assert str(caught.value) == "the open-issue count came back in an unreadable shape"


def test_no_repository_is_refused_before_anything_is_fetched(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test 20. The slug comes from GITHUB_REPOSITORY, and its absence is a named line."""
    from tools import stale_bodies

    assert stale_bodies.main(["--pull", "1", "--repo", ""]) == 2
    err = capsys.readouterr().err.strip()
    assert "set GITHUB_REPOSITORY or pass --repo" in err
    assert len(err.splitlines()) == 1


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
    """Test 15. A list left saying a merge landed is the defect this module exists to catch."""
    body = render([], state="abandoned")
    assert body is not None
    assert MARKER in body
    assert "without merging" in body


def test_an_abandoned_pull_request_with_no_comment_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 16. Telling a reader nothing happened, where the check never spoke, is an empty note."""
    from tools import stale_bodies

    written: list[list[str]] = []

    def fake_gh(args: list[str]) -> str:
        written.append(args)
        return ""  # no comments on this pull request

    monkeypatch.setattr(stale_bodies, "_gh", fake_gh)
    said = stale_bodies._upsert(REPO, 999, render([], state="abandoned"), correction_only=True)
    assert said == "nothing was posted, so there is nothing to correct"
    assert not [call for call in written if "-X" in call]


def test_an_abandoned_pull_request_with_a_comment_has_it_corrected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 17. Where a list was posted, the correction has something to correct."""
    from tools import stale_bodies

    written: list[list[str]] = []

    def fake_gh(args: list[str]) -> str:
        written.append(args)
        if "--paginate" in args:
            return json.dumps({"id": 42, "body": f"{MARKER}\nsomething"})
        return ""

    monkeypatch.setattr(stale_bodies, "_gh", fake_gh)
    said = stale_bodies._upsert(REPO, 999, render([], state="abandoned"), correction_only=True)
    assert said == "edited comment 42 in place"
    patches = [call for call in written if "PATCH" in call]
    assert len(patches) == 1
    assert "without merging" in patches[0][-1]
