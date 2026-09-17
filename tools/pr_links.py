"""Compare what a pull request body claims to close against what GitHub actually parsed.

A body that spells a closing keyword in a form GitHub's body parser rejects merges without
complaint and loses the link between the work and the issue that asked for it. Sixteen of
this repository's first 182 pull requests lost or partly lost their link that way, about one
in eleven, starting at #112. Ten wrote ``` `Closes #NN` ``` inside a code span, two split the
keyword from the number across a line break, and one used no keyword at all.

**The body is not where the answer lives, which is why this asks GitHub rather than linting
the text.** #377 wrote two keywords the same way in one sentence, both inside code spans, and
GitHub parsed one of them. The one that parsed did so because its number happened to appear
in plain text elsewhere in the body. A reader checking the body sees a keyword and a number
and cannot tell which of the two took. Only ``closingIssuesReferences`` can.

**What is lost is the attribution, not usually the close.** GitHub runs a second, more
permissive keyword parser over the merge commit message, and this repository's ruleset allows
squash merges only, so every merge copies the body into that message verbatim. #294 and #368
both closed within two seconds of their pull request merging, and both record a bare
``Commit`` as the closer rather than the pull request that did the work.

The scan below is deliberately more permissive than GitHub's parser. Its job is to recover
what the author *meant*, so it reads through code spans, markdown links and a single line
break. GitHub's answer stays the ground truth for what actually happened, and a disagreement
between the two is the defect.

Three rules decide the verdict, and each carries what it measured over all 182 bodies.

1. A claimed reference missing from the parsed set **fails**. Twelve true hits, zero false.
2. A reference written ``Part of`` that GitHub parsed as closing **reports only**. Zero true
   hits, and its single firing is #147, which quotes a commit trailer in prose. A failing
   gate there would refuse a correct pull request for a defect that has never occurred.
3. A pull request claiming nothing and parsing nothing **says nothing**. Eighty-one of the
   182 link no issue at all, so a check demanding a link on every pull request would refuse
   nearly half of them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

MARKER = "<!-- pr-links-check -->"
"""Identifies this check's own comment so a second run edits rather than appends."""

_KEYWORDS = r"close[sd]?|fix(?:e[sd])?|resolve[sd]?"

_GAP = r"[ \t]*:?[ \t]*\n?[ \t]*"
"""What may sit between a keyword and its reference.

At most **one** newline and never a blank line. Both halves are measured. Allowing one
newline is what catches #364 and #372, whose keyword and number sit on adjacent lines.
Refusing a blank line is what stops #194, whose ``## How #184 was resolved`` heading is
followed by a blank line and then a link. A gap that crosses blank lines reads that heading
as a claim and fails a correct pull request.
"""

_REPO = r"(?:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+))?"
"""An optional ``owner/repo`` prefix, so a foreign number is not read as a local claim.

The Dependabot bump #31 carries fifteen bare references such as ``#2474``, every one an
upstream issue number. None of the fifteen follows a keyword, so nothing there is claimed
today. This prefix is what catches the case when that stops holding.
"""

_CLOSES = re.compile(rf"\b(?:{_KEYWORDS})\b{_GAP}[`\[]*{_REPO}#(\d+)", re.IGNORECASE)
_PART_OF = re.compile(rf"\bpart of\b{_GAP}[`\[]*{_REPO}#(\d+)", re.IGNORECASE)


class CouldNotRun(Exception):
    """The check never reached a verdict.

    Distinct from a failing verdict on purpose. Both surface as one red job, so only the
    comment can tell an author that their link is fine and the API was not.
    """


@dataclass(frozen=True)
class Claims:
    """What a body says, read permissively."""

    closes: frozenset[int] = field(default_factory=frozenset)
    part_of: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Verdict:
    """What the body says set against what GitHub parsed."""

    missing: tuple[int, ...]
    """Claimed as closing and absent from the parsed set. This is what fails the check."""

    contradicted: tuple[int, ...]
    """Written ``Part of`` and parsed as closing. Reported, never failed. See rule 2."""

    unclaimed: tuple[int, ...]
    """Parsed as closing and not found by the scan, which means the scan is too narrow."""

    parsed: tuple[int, ...]
    claimed: tuple[int, ...]

    part_of_seen: bool = False
    """Whether the body wrote ``Part of`` at all, which counts as having something to say."""

    @property
    def broken(self) -> bool:
        return bool(self.missing)

    @property
    def silent(self) -> bool:
        """Nothing claimed and nothing parsed, so there is no verdict to report."""
        return not self.claimed and not self.parsed and not self.part_of_seen


def _local(match: re.Match[str], repo: str) -> int | None:
    """The issue number when a match names this repository, and ``None`` when it does not."""
    owner, name, number = match.group(1), match.group(2), int(match.group(3))
    if owner is None:
        return number
    return number if f"{owner}/{name}".lower() == repo.lower() else None


def scan(body: str | None, repo: str) -> Claims:
    """Read a body for the references it means to close and the ones it calls partial."""
    text = body or ""
    closes = {n for n in (_local(m, repo) for m in _CLOSES.finditer(text)) if n is not None}
    part_of = {n for n in (_local(m, repo) for m in _PART_OF.finditer(text)) if n is not None}
    return Claims(frozenset(closes), frozenset(part_of))


def parsed_numbers(references: list[dict], repo: str) -> frozenset[int]:
    """The issue numbers GitHub parsed, narrowed to this repository.

    ``closingIssuesReferences`` carries each reference's own repository, so a cross-repository
    close cannot collide with a local issue that happens to share its number.
    """
    owner, _, name = repo.partition("/")
    here = set()
    for ref in references:
        holder = ref.get("repository") or {}
        holder_owner = (holder.get("owner") or {}).get("login", "")
        if holder_owner.lower() == owner.lower() and holder.get("name", "").lower() == name.lower():
            here.add(int(ref["number"]))
    return frozenset(here)


def assess(claims: Claims, parsed: frozenset[int]) -> Verdict:
    """Set the body's claims against GitHub's answer."""
    return Verdict(
        missing=tuple(sorted(claims.closes - parsed)),
        contradicted=tuple(sorted(claims.part_of & parsed)),
        unclaimed=tuple(sorted(parsed - claims.closes)),
        parsed=tuple(sorted(parsed)),
        claimed=tuple(sorted(claims.closes)),
        part_of_seen=bool(claims.part_of),
    )


def _refs(numbers: tuple[int, ...]) -> str:
    return ", ".join(f"#{n}" for n in numbers) if numbers else "nothing"


def render(verdict: Verdict, repo: str, pr: int) -> str:
    """The comment body.

    The failure message names the repair rather than restating the rule, and it names the
    command to re-verify with. Reading the body back does not work, because the body is not
    where the answer is.
    """
    lines = [MARKER, ""]
    if verdict.broken:
        lines += [
            "## This pull request's closing link did not parse",
            "",
            f"The body claims to close {_refs(verdict.claimed)}. GitHub parsed "
            f"{_refs(verdict.parsed)}. Missing: **{_refs(verdict.missing)}**.",
            "",
            "A keyword inside a code span, inside a markdown link split across a line break, "
            "or with the number on the next line does not parse. Write it as plain text with "
            "the number immediately after the keyword, on one line, then re-verify:",
            "",
            "```bash",
            f"gh pr view {pr} --repo {repo} --json closingIssuesReferences",
            "```",
            "",
            "Reading the body back does not tell you. Only that command does.",
        ]
    elif verdict.silent:
        lines += [
            "## This pull request closes nothing",
            "",
            "Its body claims no closing keyword and GitHub parsed none, which is the ordinary "
            "shape for a partial fix or a change with no issue behind it. Nothing here is "
            "wrong. This note is only replacing an earlier verdict that no longer holds.",
        ]
    else:
        lines += [
            "## Closing links parsed",
            "",
            f"GitHub parsed this pull request as closing {_refs(verdict.parsed)}.",
        ]
    if verdict.contradicted:
        lines += [
            "",
            f"Also worth a look: {_refs(verdict.contradicted)} "
            "is written `Part of` and GitHub parsed it as closing. That is fine when the body "
            "quotes the phrase rather than using it, which is the only case seen so far.",
        ]
    if verdict.unclaimed:
        lines += [
            "",
            f"GitHub parsed {_refs(verdict.unclaimed)} without this check finding a keyword "
            "for it. That means the scan is narrower than GitHub's parser, which is a defect "
            "in the check rather than in this body.",
        ]
    return "\n".join(lines) + "\n"


def render_could_not_run(reason: str, pr: int, repo: str) -> str:
    """Say the check failed rather than letting a red job read as a lost link."""
    return (
        f"{MARKER}\n\n"
        "## This check could not run\n\n"
        "It reached no verdict, so this says nothing about whether the closing link parsed. "
        f"Check it by hand:\n\n```bash\ngh pr view {pr} --repo {repo} "
        "--json closingIssuesReferences\n```\n\n"
        f"The query failed with: `{reason}`\n"
    )


def _gh(args: list[str]) -> str:
    try:
        done = subprocess.run(
            ["gh", *args], capture_output=True, text=True, check=False, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CouldNotRun(f"gh could not be run: {exc}") from exc
    if done.returncode != 0:
        raise CouldNotRun(f"gh exited {done.returncode}: {done.stderr.strip()[:300]}")
    return done.stdout


def fetch(pr: int, repo: str) -> tuple[str | None, list[dict]]:
    """The body and the parsed references, in one query.

    The body is read from the API rather than from the workflow's event payload. That keeps
    it out of every shell expression, which is what the repository's code scanning setup
    flags, and it also means an edit made seconds ago and the parse of that edit come from
    the same read.
    """
    raw = _gh(["pr", "view", str(pr), "--repo", repo, "--json", "body,closingIssuesReferences"])
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CouldNotRun(f"gh returned something that is not JSON: {exc}") from exc
    return payload.get("body"), payload.get("closingIssuesReferences") or []


def _existing_comment(pr: int, repo: str) -> int | None:
    raw = _gh(["api", f"repos/{repo}/issues/{pr}/comments", "--paginate"])
    try:
        comments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CouldNotRun(f"comment listing is not JSON: {exc}") from exc
    for comment in comments:
        if MARKER in (comment.get("body") or ""):
            return int(comment["id"])
    return None


def publish(body: str, pr: int, repo: str, *, only_if_present: bool) -> str:
    """Write the verdict where the author is already reading.

    ``only_if_present`` carries the quiet rule. A pull request with no verdict gets no new
    comment, and that is also what keeps the check silent on Dependabot bumps, whose token
    cannot write one anyway. The second run is the exception: a comment that already exists
    is rewritten whatever the verdict, so an author who repairs a body is not left with the
    first run's failure still sitting there.
    """
    existing = _existing_comment(pr, repo)
    if existing is None:
        if only_if_present:
            return "stayed quiet"
        _gh(["api", f"repos/{repo}/issues/{pr}/comments", "-f", f"body={body}"])
        return "commented"
    _gh(
        [
            "api",
            "--method",
            "PATCH",
            f"repos/{repo}/issues/comments/{existing}",
            "-f",
            f"body={body}",
        ]
    )
    return "updated"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not args or not repo:
        print("usage: GITHUB_REPOSITORY=owner/name python -m tools.pr_links <pr-number>")
        return 2
    pr = int(args[0])

    try:
        body, references = fetch(pr, repo)
    except CouldNotRun as exc:
        print(f"could not run: {exc}", file=sys.stderr)
        try:
            publish(render_could_not_run(str(exc), pr, repo), pr, repo, only_if_present=False)
        except CouldNotRun as second:
            print(f"could not say so either: {second}", file=sys.stderr)
        return 2

    verdict = assess(scan(body, repo), parsed_numbers(references, repo))
    comment = render(verdict, repo, pr)
    print(f"claimed {_refs(verdict.claimed)}; GitHub parsed {_refs(verdict.parsed)}")

    try:
        print(publish(comment, pr, repo, only_if_present=verdict.silent))
    except CouldNotRun as exc:
        print(f"the verdict stands, the comment did not: {exc}", file=sys.stderr)

    if verdict.broken:
        print(f"::error::closing keyword did not parse for {_refs(verdict.missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
