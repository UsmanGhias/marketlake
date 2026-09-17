"""Name the open issue bodies a merge just made worth re-reading.

A merge can falsify an open issue's body without anyone noticing, because the pull
request and the issue it closed both read correctly and only the third document, which
described the gap in the present tense, is now wrong. This module names the candidates
at merge time. It does not decide whether a body is stale, which is a judgment. It says
which bodies just became worth re-reading, which is a lookup.

Two lookups, unioned. The first takes the issues the pull request's own body names,
minus the ones it closes. The second takes the open issues whose bodies cite an issue
the pull request closes. Neither alone covers the six cases that produced #393.

The matching is a pure function over fetched JSON and the fetch is a thin command around
it. That split is what lets the tests run under ``tests/conftest.py``'s network guard,
which fails any test that reaches another machine.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "MARKER",
    "Candidate",
    "StaleBodiesError",
    "candidates",
    "main",
    "references",
    "render",
]

MARKER = "<!-- stale-body-candidates -->"

# A bare ``#123``. The lookbehind drops ``actions/checkout#2454``, the foreign form that
# 45 of this repo's 46 such references take, all inside one Dependabot body. It also
# drops ``l3a0/marketlake#58``, which is real, so _SELF below puts that one back.
_HASH = re.compile(r"(?<![\w/-])#(\d+)\b")

# Reference forms this repo does not use, and the reason each is left unmatched, are in
# #393: ``pull/NNN`` URLs are safe to ignore because issues and pull requests share one
# number sequence, and ``GH-NNN`` appears zero times in 413 bodies.


class StaleBodiesError(RuntimeError):
    """A refusal an operator reads as one line, not a traceback."""


@dataclass(frozen=True)
class Candidate:
    """One open issue, with every reason it was named."""

    number: int
    title: str
    reasons: tuple[str, ...]


def _self_pattern(repo: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(repo)}#(\d+)\b")


def _url_pattern(repo: str) -> re.Pattern[str]:
    return re.compile(rf"github\.com/{re.escape(repo)}/issues/(\d+)")


def references(body: str | None, repo: str) -> set[int]:
    """Every issue number ``body`` names, in the three forms this repo writes.

    A pull request opened with no body arrives as a null rather than an empty string,
    and that names nothing. Code fences and inline code spans are read rather than
    stripped: every one of the six fenced references in this repo is a real reference,
    so stripping would lose them.
    """
    if not body:
        return set()
    found = {int(m) for m in _HASH.findall(body)}
    found |= {int(m) for m in _self_pattern(repo).findall(body)}
    found |= {int(m) for m in _url_pattern(repo).findall(body)}
    return found


def candidates(
    *,
    pull_body: str | None,
    closes: Iterable[int],
    issues: Iterable[dict[str, object]],
    repo: str,
) -> list[Candidate]:
    """The open issues this merge makes worth re-reading, most recent first.

    ``issues`` carries each issue's ``number``, ``title``, ``body`` and ``state``. The
    state is checked here as well as in the fetch, because "a closed issue is never
    named" is a contract this function owes its caller rather than a property of a query
    string.
    """
    closing = set(closes)
    named = references(pull_body, repo) - closing
    found: dict[int, list[str]] = {}
    for issue in issues:
        number = int(issue["number"])  # type: ignore[arg-type]
        if number in closing:
            continue
        if str(issue.get("state", "OPEN")).upper() != "OPEN":
            continue
        reasons: list[str] = []
        if number in named:
            reasons.append("this pull request's body names it")
        cited = sorted(references(issue.get("body"), repo) & closing)  # type: ignore[arg-type]
        if cited:
            joined = ", ".join(f"#{n}" for n in cited)
            reasons.append(f"cites {joined}, which this closes")
        if reasons:
            found[number] = reasons
    titles = {int(i["number"]): str(i.get("title", "")) for i in issues}  # type: ignore[arg-type]
    return [
        Candidate(number=n, title=titles.get(n, ""), reasons=tuple(found[n]))
        for n in sorted(found, reverse=True)
    ]


def _escape(text: str) -> str:
    """Keep a title inside its table cell.

    No title in this tracker carries a pipe today, so this is precaution rather than
    repair, and it costs one call.
    """
    return text.replace("|", "\\|")


def render(found: Sequence[Candidate], *, state: str) -> str | None:
    """The comment body, or ``None`` when there is nothing to say.

    ``state`` is ``open``, ``merged`` or ``abandoned``. An empty set on an open pull
    request renders nothing at all. An empty set after a merge, or a pull request closed
    without merging, renders a correction, because a list left standing that says a merge
    landed is the defect this module exists to catch.
    """
    if state == "abandoned":
        return (
            f"{MARKER}\n### Bodies to re-read\n\n"
            "This pull request closed without merging, so nothing landed and no body "
            "changed under anyone.\n"
        )
    if not found:
        return None
    heading = (
        "Bodies worth re-reading after this merge"
        if state == "merged"
        else "Bodies to re-read when this merges"
    )
    lines = [
        MARKER,
        f"### {heading}",
        "",
        "Each of these open issues just became worth a second read. This does not say any of",
        "them is stale, which is a judgment. It says which bodies to look at, which is a lookup.",
        "",
        "| Issue | Why | Title |",
        "| --- | --- | --- |",
    ]
    lines += [f"| #{c.number} | {'; '.join(c.reasons)} | {_escape(c.title)} |" for c in found]
    lines.append("")
    return "\n".join(lines)


def _gh(args: Sequence[str]) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        tail = detail[-1] if detail else f"exit {result.returncode}"
        raise StaleBodiesError(f"gh {' '.join(args)} failed: {tail}")
    return result.stdout


def _fetch_closes(repo: str, number: int) -> list[int]:
    """What GitHub parsed as this pull request's closing references.

    The webhook payload cannot supply this. ``gh api repos/<repo>/pulls/<n>`` carries no
    closing-reference key, because the field is GraphQL only.
    """
    raw = _gh(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "closingIssuesReferences",
        ]
    )
    payload = json.loads(raw)
    return [int(ref["number"]) for ref in payload.get("closingIssuesReferences", [])]


_LIMIT = 2000


def _list_open_issues(repo: str) -> list[dict[str, object]]:
    raw = _gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            str(_LIMIT),
            "--json",
            "number,title,body,state",
        ]
    )
    return list(json.loads(raw))


def _count_open_issues(repo: str) -> int:
    owner, _, name = repo.partition("/")
    owner_literal, name_literal = json.dumps(owner), json.dumps(name)
    query = (
        f"{{repository(owner:{owner_literal},name:{name_literal})"
        "{issues(states:OPEN){totalCount}}}"
    )
    total = json.loads(_gh(["api", "graphql", "-f", f"query={query}"]))
    return int(total["data"]["repository"]["issues"]["totalCount"])


def _fetch_open_issues(repo: str) -> list[dict[str, object]]:
    """Every open issue, or a refusal.

    A short list is the failure this module exists to prevent, arriving from inside the
    module, so the count is checked twice. Against the limit, which catches a cap:
    ``gh issue list --limit 5`` returns five and says nothing about the rest. And against
    ``totalCount``, which catches a short page.

    The second check races, and the race is not hypothetical. This module's own first run
    refused with "returned 111 of 112" because another session filed an issue between the
    listing and the count. A guard that refuses whenever somebody files an issue is a
    guard whose reader turns it off, which is the failure this whole module is built
    around. So a shortfall is read once more before it is believed. A race resolves on the
    second attempt, because the new issue is in the second listing. A genuinely short page
    repeats.
    """
    shortfall = ""
    for _ in range(2):
        issues = _list_open_issues(repo)
        if len(issues) >= _LIMIT:
            raise StaleBodiesError(
                f"the open-issue listing hit its {_LIMIT} limit, so it is truncated and "
                "would drop candidates silently"
            )
        expected = _count_open_issues(repo)
        if len(issues) >= expected:
            return issues
        shortfall = (
            f"the open-issue listing returned {len(issues)} of {expected} twice, so it "
            "is short and would drop candidates silently"
        )
    raise StaleBodiesError(shortfall)


def _find_marker_comment(repo: str, number: int) -> int | None:
    """This run's own earlier comment, so a later run edits rather than adds.

    ``--jq`` emits one object per line, which is what makes ``--paginate`` readable: the
    raw form concatenates a JSON array per page.
    """
    raw = _gh(
        [
            "api",
            "--paginate",
            f"repos/{repo}/issues/{number}/comments",
            "--jq",
            ".[] | {id: .id, body: .body}",
        ]
    )
    for line in raw.splitlines():
        if not line.strip():
            continue
        comment = json.loads(line)
        if MARKER in (comment.get("body") or ""):
            return int(comment["id"])
    return None


def _upsert(repo: str, number: int, body: str | None) -> str:
    existing = _find_marker_comment(repo, number)
    if body is None:
        if existing is None:
            return "nothing to say, and no comment to correct"
        _gh(
            [
                "api",
                "-X",
                "PATCH",
                f"repos/{repo}/issues/comments/{existing}",
                "-f",
                f"body={MARKER}\n### Bodies to re-read\n\nNothing here now names a body "
                "worth re-reading.\n",
            ]
        )
        return f"corrected comment {existing} to say the list is empty"
    if existing is None:
        _gh(["api", "-X", "POST", f"repos/{repo}/issues/{number}/comments", "-f", f"body={body}"])
        return "posted the candidate list"
    _gh(["api", "-X", "PATCH", f"repos/{repo}/issues/comments/{existing}", "-f", f"body={body}"])
    return f"edited comment {existing} in place"


def main(argv: Sequence[str] | None = None) -> int:
    """Name the candidates for one pull request and write them to its comment."""
    parser = argparse.ArgumentParser(
        prog="stale_bodies",
        description="Name the open issue bodies a merge makes worth re-reading.",
    )
    parser.add_argument("--pull", type=int, required=True, help="the pull request number")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="owner/name, defaulting to GITHUB_REPOSITORY",
    )
    parser.add_argument(
        "--state",
        choices=("open", "merged", "abandoned"),
        default="open",
        help="where the pull request stands",
    )
    parser.add_argument("--body-file", help="a file holding the pull request's body")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the comment instead of writing it"
    )
    args = parser.parse_args(argv)

    if not args.repo or "/" not in args.repo:
        print(
            "stale_bodies: no repository; set GITHUB_REPOSITORY or pass --repo owner/name",
            file=sys.stderr,
        )
        return 2

    try:
        if args.state == "abandoned":
            body = render([], state="abandoned")
        else:
            pull_body = None
            if args.body_file:
                with open(args.body_file, encoding="utf-8") as handle:
                    pull_body = handle.read()
            closes = _fetch_closes(args.repo, args.pull)
            issues = _fetch_open_issues(args.repo)
            found = candidates(pull_body=pull_body, closes=closes, issues=issues, repo=args.repo)
            body = render(found, state=args.state)
        if args.dry_run:
            print(body if body is not None else "stale_bodies: nothing to re-read")
            return 0
        print(f"stale_bodies: {_upsert(args.repo, args.pull, body)}")
    except StaleBodiesError as error:
        print(f"stale_bodies: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
