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

Four rules decide the verdict, and each carries what it measured over all 182 bodies.

1. A claimed reference missing from the parsed set **fails**. Eleven true hits and one false.
2. A claim on an issue that **closed before this pull request existed** reports rather than
   fails. That one rule is what makes rule 1 eleven-for-eleven instead of twelve-for-twelve.
   #112 narrates #85's body fifteen hours after #77 had already closed, quoting the string
   ``` `Closes #77.` ``` inside a numbered list. A closed issue cannot be closed again, so
   nothing is lost there and GitHub parsing nothing is correct.
3. A reference written ``Part of`` that GitHub parsed as closing **reports only**. Zero true
   hits, and its single firing is #147, which quotes a commit trailer in prose. A failing
   gate there would refuse a correct pull request for a defect that has never occurred.
4. A pull request claiming nothing and parsing nothing **says nothing**. Eighty-one of the
   182 link no issue at all, so a check demanding a link on every pull request would refuse
   nearly half of them.

The scan cannot tell a claim from a quotation, and this repository quotes closing keywords
constantly. Rule 2 removes the one case the corpus actually contains. The rest is named as a
limit rather than solved: write an example without a ``#`` where one is needed.
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

_GAP = r"[ \t]*+:?+[ \t]*+(?:\r?\n)?+[ \t]*+"
"""What may sit between a keyword and its reference.

**Every quantifier here is possessive, and that is load-bearing rather than tidy.** Three
plain ``[ \t]*`` groups over one character class made the scan cubic in the length of a
whitespace run, because the engine tries every way of splitting that run between them. A
keyword followed by spaces and no ``#`` is all it takes, which anyone opening a pull request
can write and re-trigger at will through the ``edited`` type. Measured before the fix: 400
spaces took 0.86s, 800 took 6.8s and 1200 took 22.7s, so the job's five-minute ceiling
arrives around 3,000 characters and a body may hold 65,536. A hang reaches none of the
:class:`CouldNotRun` machinery either. It is a killed job, a red check, no comment, and
whatever the last run said left standing beside it. Possessive quantifiers never give back
what they matched, so a run that cannot reach a ``#`` fails at once.

At most **one** line ending and never a blank line. Both halves are measured. Allowing one
newline is what catches #364 and #372, whose keyword and number sit on adjacent lines.
Refusing a blank line is what stops #194, whose ``## How #184 was resolved`` heading is
followed by a blank line and then a link. A gap that crosses blank lines reads that heading
as a claim and fails a correct pull request.

The ``\r`` is not decoration. A body edited through the web form comes back with CRLF line
endings, because that is what an HTML textarea submits, and every body in this repository's
corpus came from ``gh`` with LF. Without it, the one spelling the ``edited`` trigger exists
to re-check is the one spelling the scan goes blind to, and a blind scan reports that
nothing is wrong.
"""

_REPO = r"(?:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+))?"
"""An optional ``owner/repo`` prefix, so a foreign number is not read as a local claim.

The Dependabot bump #31 carries fifteen bare references such as ``#2474``, every one an
upstream issue number. None of the fifteen follows a keyword, so nothing there is claimed
today. This prefix is what catches the case when that stops holding.
"""

_URL = r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/issues/"
"""A full issue URL, which GitHub parses and a bare ``#NN`` scan does not see.

Zero of the 182 bodies write a keyword in front of one, but 65 of them carry an issue URL
somewhere, so the vocabulary is already in the habit. Inside a code span GitHub rejects it
and a scan that cannot see it reports that nothing is wrong, which is the original defect
with the alarm switched off. Adding this form introduced no new failure across the corpus.
"""

_REF = rf"(?:{_REPO}#|GH-|{_URL})(\d+)"
"""Every spelling of a reference: bare, ``owner/repo#NN``, ``GH-NN`` and a full issue URL."""

_DECOR = r"(?:[`(*_]|\[[^\]\n#]{1,60}\]\(|\[)*+"
"""Leading decoration the house style puts in front of a reference.

The two branches beginning with ``[`` must not overlap, and that is what keeps this linear.
A titled link's label is required to hold no ``#``, so ``[the issue](...)`` takes the link
branch while ``[#390](...)`` and ``[l3a0/marketlake#390](...)`` take the bare-bracket branch
and let the reference itself match. Without that split, the link branch would swallow
``[#390](`` and a possessive star could not give it back.

The possessive quantifier is belt-and-braces rather than load-bearing, and it is worth being
straight about which. Measured both ways at `d5cadf3`, a plain star is already linear here:
5,000 brackets take 0.0023s against 0.0017s possessive. So no test distinguishes the two, and
mutation testing reports this one surviving. It stays because the disjointness it depends on
lives in a different expression, :data:`_URL` and the label class above, and a later edit
widening either would reintroduce the overlap silently. :data:`_GAP` is the opposite case,
where the quantifier is the whole defence.

Backticks, brackets and emphasis, plus a markdown link whose label is words rather than the
number, as in ``Closes [the issue](.../issues/390)``. GitHub parses that one and a scan
stopping at the bracket does not, so without the label form the check reads a real claim as
no claim at all.
"""

_CLOSES = re.compile(rf"\b(?:{_KEYWORDS})\b{_GAP}{_DECOR}{_REF}", re.IGNORECASE)
_PART_OF = re.compile(rf"\bpart of\b{_GAP}{_DECOR}{_REF}", re.IGNORECASE)


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

    stale: tuple[int, ...]
    """Claimed, unparsed, and already closed before this pull request existed.

    Reported rather than failed. See rule 2 in the module docstring: a closed issue cannot be
    closed again, so the body is narrating rather than claiming, and #112 is the case.
    """

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
    """The issue number when a match names this repository, and ``None`` when it does not.

    The pattern carries two optional owner and name pairs, one from the ``owner/repo#NN``
    prefix and one from a full issue URL, so whichever matched is the one to compare. Both
    sides are lowered, because GitHub treats an owner and a repository name as
    case-insensitive and ``L3a0/Marketlake#390`` is the same issue.
    """
    prefix_owner, prefix_name, url_owner, url_name = match.group(1, 2, 3, 4)
    owner, name = (prefix_owner, prefix_name) if prefix_owner else (url_owner, url_name)
    number = int(match.group(5))
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
        holder_owner = (holder.get("owner") or {}).get("login") or ""
        holder_name = holder.get("name") or ""
        if holder_owner.lower() == owner.lower() and holder_name.lower() == name.lower():
            number = ref.get("number")
            if number is not None:
                here.add(int(number))
    return frozenset(here)


def assess(
    claims: Claims, parsed: frozenset[int], already_closed: frozenset[int] = frozenset()
) -> Verdict:
    """Set the body's claims against GitHub's answer.

    ``already_closed`` holds the claimed issues that closed before this pull request existed.
    They move out of ``missing`` and into ``stale``, which reports and does not fail.
    """
    unparsed = claims.closes - parsed
    return Verdict(
        missing=tuple(sorted(unparsed - already_closed)),
        stale=tuple(sorted(unparsed & already_closed)),
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
    if verdict.stale:
        lines += [
            "",
            f"The body also names {_refs(verdict.stale)} after a closing keyword, and GitHub "
            "did not parse that either. It is reported rather than failed because "
            f"{'each of those issues' if len(verdict.stale) > 1 else 'that issue'} had already "
            "closed before this pull request existed, so nothing is lost. A body quoting "
            "another pull request's text reads this way.",
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


def fetch(pr: int, repo: str) -> tuple[str | None, list[dict], str]:
    """The body, the parsed references and the creation time, in one query.

    The body is read from the API rather than from the workflow's event payload. That keeps
    it out of every shell expression, which is what the repository's code scanning setup
    flags, and it also means an edit made seconds ago and the parse of that edit come from
    the same read.
    """
    raw = _gh(
        ["pr", "view", str(pr), "--repo", repo, "--json", "body,closingIssuesReferences,createdAt"]
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CouldNotRun(f"gh returned something that is not JSON: {exc}") from exc
    return (
        payload.get("body"),
        payload.get("closingIssuesReferences") or [],
        payload.get("createdAt", ""),
    )


def closed_before(numbers: tuple[int, ...], when: str, repo: str) -> frozenset[int]:
    """Which of these issues had already closed when the pull request was created.

    Only ever asked about claims that did not parse, so it costs a query on the failure path
    and nothing on the ordinary one. A query that fails returns nothing rather than raising,
    which leaves every claim in ``missing``. Failing loudly on an unproven claim is the safe
    direction: the worst case is the false alarm this rule exists to remove, and the comment
    says enough for a reader to dismiss it.
    """
    if not numbers or not when:
        return frozenset()
    owner, _, name = repo.partition("/")
    fields = " ".join(f"i{n}: issue(number: {n}) {{ number closedAt }}" for n in numbers)
    query = f'{{ repository(owner: "{owner}", name: "{name}") {{ {fields} }} }}'
    try:
        payload = json.loads(_gh(["api", "graphql", "-f", f"query={query}"]))
    except (CouldNotRun, json.JSONDecodeError, KeyError, TypeError):
        return frozenset()
    try:
        issues = ((payload.get("data") or {}).get("repository") or {}).values()
        return frozenset(
            issue["number"]
            for issue in issues
            if issue and issue.get("closedAt") and issue["closedAt"] < when
        )
    except (AttributeError, KeyError, TypeError):
        return frozenset()


AUTHOR = "github-actions[bot]"
"""The only author whose comment this check will edit.

The marker alone is not enough to identify its own comment. A person quoting the check's
output in raw markdown carries the marker too, and a listing comes back oldest first, so the
quote would be found first, overwritten with a verdict, and then overwritten again on every
later run while the check's real comment went stale. Repository write access is enough to
edit anyone's comment, so nothing but this filter stops it.
"""


def _existing_comment(pr: int, repo: str) -> int | None:
    raw = _gh(["api", f"repos/{repo}/issues/{pr}/comments", "--paginate"])
    try:
        comments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CouldNotRun(f"comment listing is not JSON: {exc}") from exc
    try:
        for comment in comments:
            if MARKER not in (comment.get("body") or ""):
                continue
            if (comment.get("user") or {}).get("login") != AUTHOR:
                continue
            return int(comment["id"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise CouldNotRun(f"the comment listing is not shaped like comments: {exc}") from exc
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
    if not args or "/" not in repo:
        print("usage: GITHUB_REPOSITORY=owner/name python -m tools.pr_links <pr-number>")
        return 2
    try:
        pr = int(args[0])
    except ValueError:
        print(f"not a pull request number: {args[0]!r}", file=sys.stderr)
        return 2

    try:
        body, references, created_at = fetch(pr, repo)
    except CouldNotRun as exc:
        print(f"could not run: {exc}", file=sys.stderr)
        try:
            publish(render_could_not_run(str(exc), pr, repo), pr, repo, only_if_present=False)
        except CouldNotRun as second:
            print(f"could not say so either: {second}", file=sys.stderr)
        return 2

    claims = scan(body, repo)
    parsed = parsed_numbers(references, repo)
    unparsed = tuple(sorted(claims.closes - parsed))
    verdict = assess(claims, parsed, closed_before(unparsed, created_at, repo))
    comment = render(verdict, repo, pr)
    print(f"claimed {_refs(verdict.claimed)}; GitHub parsed {_refs(verdict.parsed)}")
    if verdict.stale:
        print(f"already closed before this pull request existed: {_refs(verdict.stale)}")

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
