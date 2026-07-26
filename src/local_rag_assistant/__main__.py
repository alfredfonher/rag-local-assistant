"""Module entry point for ``python -m local_rag_assistant``."""

from __future__ import annotations

from collections.abc import Sequence

from local_rag_assistant.cli.commands import main as cli_main


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the installed CLI entry point."""
    args = list(argv) if argv is not None else None
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
