"""Console entry point: python -m intentgraph / the `intentgraph` script."""

from __future__ import annotations

import sys


def main() -> None:
    from intentgraph.cli import main as _main
    raise SystemExit(_main())


if __name__ == "__main__":
    main()
