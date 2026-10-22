"""JSON command line entry point for the structured PESS solver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import solve_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve a structured PESS instance")
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="JSON request file; use - or omit to read stdin",
    )
    args = parser.parse_args()
    try:
        raw = (
            sys.stdin.read()
            if args.input == "-"
            else Path(args.input).read_text(encoding="utf-8")
        )
        response = solve_payload(json.loads(raw))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"pess-ess-solve: {exc}", file=sys.stderr)
        return 2
    json.dump(response, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
