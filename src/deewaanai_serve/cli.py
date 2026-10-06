# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Mug and Bewong Ltd (Company No. 14916888).
# DeeWaanAI(TM) is a trademark of Mug and Bewong Ltd. See LICENSE.
"""dwa — DeeWaanAI Serve command-line interface.

Subcommands:
  dwa serve    Run the pinned-expert OpenAI-compatible server (MLX, Apple Silicon).
  dwa convert  Build pinned-engine expert slabs from an MLX 4-bit checkpoint (verbatim).

Run ``dwa <subcommand> --help`` for full options.
"""
from __future__ import annotations

import sys

USAGE = """dwa — DeeWaanAI Serve (pinned-expert MoE serving on Apple Silicon)

usage:
  dwa serve    --ref-dir <mlx-4bit-model> --experts-dir <slabs> [--port 8080 --pin-fraction 1.0 ...]
  dwa convert  --mlx-dir <mlx-4bit-model> --out <experts-dir>

  dwa serve --help | dwa convert --help     # full options
"""


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    sys.argv = [f"dwa {cmd}", *rest]  # hand the remaining args to the target main()
    if cmd == "serve":
        from .server import main as _m
    elif cmd == "convert":
        from .convert import main as _m
    else:
        sys.stderr.write(f"dwa: unknown command '{cmd}'\n\n{USAGE}")
        return 2
    result = _m()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
