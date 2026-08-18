# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dev-buddy MCP — universal, anywhere-runnable developer tools.

Toy. The author's real dev-helper has 20+ tools (stash, rebase, gh/glab,
jq, ...). Pattern scales; start here. GitLab/GitHub platform-specific
tools are deliberately not bundled — they're the natural first
"grow-your-own" exercise.
"""
from __future__ import annotations

import pathlib
import shlex
import subprocess
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

from _git_whitelist import TIER1, TIER2, _strip_escape_flags, _validate_tier2

mcp = FastMCP("dev-buddy")


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout
    if result.stderr:
        out = f"{out}\n[stderr]\n{result.stderr}" if out else result.stderr
    if result.returncode != 0:
        out = f"[exit {result.returncode}]\n{out}"
    return out or "(no output)"


def _run_with_stdin(cmd: list[str], text: str) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, input=text)
    out = result.stdout
    if result.stderr:
        out = f"{out}\n[stderr]\n{result.stderr}" if out else result.stderr
    if result.returncode != 0:
        out = f"[exit {result.returncode}]\n{out}"
    return out or "(no output)"


@mcp.tool()
def git_read(op: str, args: str = "") -> str:
    """Run a read-only git command. Allowed ops + arg shapes are defined
    in mcp-servers/_git_whitelist.py. Global escape-hatch flags (-c, -C,
    --git-dir, --work-tree, --namespace, --exec-path, --config-env) are
    stripped before exec."""
    argv = _strip_escape_flags(shlex.split(args) if args else [])
    if op in TIER1:
        return _run(["git", op, *argv])
    if op in TIER2:
        if not _validate_tier2(TIER2[op], argv):
            return (f"[error] '{op} {args}' not in tier-2 allow-list. "
                    f"Why: {TIER2[op].get('why', '(none)')}")
        return _run(["git", op, *argv])
    return (f"[error] op '{op}' not whitelisted "
            f"(see mcp-servers/_git_whitelist.py).")


@mcp.tool()
def git_add_update(paths: str = "") -> str:
    """Stage changes to tracked files (`git add --update [paths]`). Never adds new files."""
    cmd = ["git", "add", "--update"]
    if paths:
        cmd.extend(shlex.split(paths))
    return _run(cmd)


@mcp.tool()
def run_pytest(target: str = "", extra_args: str = "") -> str:
    """Run pytest against target (file/dir/nodeid); extra_args appended verbatim."""
    cmd = ["pytest"]
    if target:
        cmd.append(target)
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return _run(cmd)


@mcp.tool()
def run_precommit(all_files: bool = False) -> str:
    """Run pre-commit on staged changes, or on all files when all_files=True."""
    cmd = ["pre-commit", "run"]
    if all_files:
        cmd.append("--all-files")
    return _run(cmd)


@mcp.tool()
def now() -> str:
    """Return current wall-clock time in ISO-8601 (UTC)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@mcp.tool()
def create_dir(path: str, parents: bool = True) -> str:
    """Create a directory. parents=True (default) is `mkdir -p`."""
    cmd = ["mkdir"]
    if parents:
        cmd.append("-p")
    cmd.append(path)
    return _run(cmd)


@mcp.tool()
def stat(path: str) -> str:
    """Return size / mtime / mode / type for path, or 'missing' if absent.
    Use Read for file contents and Glob for listings — this is just metadata."""
    p = pathlib.Path(path)
    if not p.exists():
        return f"{path}: missing"
    st = p.stat()
    kind = "dir" if p.is_dir() else ("symlink" if p.is_symlink() else "file")
    return (
        f"{path}: {kind} size={st.st_size} mode={oct(st.st_mode)[-4:]} "
        f"mtime={datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(timespec='seconds')}"
    )


@mcp.tool()
def jq(filter: str, input_file: str = "", input_text: str = "") -> str:
    """Run a jq filter. Provide exactly one of input_file or input_text."""
    if bool(input_file) == bool(input_text):
        return "[error] provide exactly one of input_file or input_text"
    if input_file:
        return _run(["jq", filter, input_file])
    return _run_with_stdin(["jq", filter], input_text)


if __name__ == "__main__":
    mcp.run()
