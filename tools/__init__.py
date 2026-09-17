"""Repository tooling that runs in CI, kept out of the ``lake`` package on purpose.

``lake`` is the market data lake. Nothing here reads or writes it, and nothing here is
imported by it. The split also keeps this package off ``lake``'s dependency set: every
module below runs on the standard library alone, so the workflows that call them skip
``uv sync`` entirely.
"""
