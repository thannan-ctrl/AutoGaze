# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared validation + command construction for script-runner and slurm-buddy."""
from __future__ import annotations

import shlex
import subprocess


def validate_script(script: str) -> None:
    """Raise ValueError unless `script` is a .py file tracked in the current git repo."""
    if not script.endswith(".py"):
        raise ValueError(f"script must end in .py, got {script!r}")
    check = subprocess.run(
        ["git", "ls-files", "--error-unmatch", script],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        raise ValueError(
            f"script is not tracked in a git repo: {check.stderr.strip()}"
        )


def build_launch_cmd(script: str, args: str = "") -> list[str]:
    """Return the argv for launching `script` with `args` under plain python."""
    return ["python", script, *shlex.split(args)]
