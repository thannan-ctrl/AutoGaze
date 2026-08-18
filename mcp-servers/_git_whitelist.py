# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Git read-only allowlist, validators, and Bash-hook CLI.

Imported as a library by dev_buddy.py; invoked as a CLI by
.claude/bash_guard.sh with one shell segment in argv[1].
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys

TIER1 = frozenset({
    "status", "log", "show", "diff", "fetch", "blame", "grep",
    "for-each-ref", "ls-remote", "ls-files", "rev-parse",
    "merge-base", "describe", "shortlog", "check-ignore",
})

TIER2 = {
    "branch": {
        "allow_bare": True,
        "safe_leading": {"flag_exact": (
            "-a", "--all", "-r", "--remotes",
            "-v", "-vv", "--verbose", "-l", "--list", "--show-current",
            "--contains", "--no-contains", "--merged", "--no-merged",
        )},
        "forbid_other_flags": True,
        "why": "Bare `git branch <name>` creates; `-d` / `-m` / `-u` mutate.",
    },
    "tag": {
        "allow_bare": False,
        "safe_leading": {
            "flag_exact": ("-l", "--list"),
            "flag_prefix": ("-n",),
        },
        "forbid_other_flags": True,
        "why": "Bare `git tag <name>` creates; `-d` deletes.",
    },
    "remote": {
        "allow_bare": False,
        "safe_leading": {
            "flag_exact": ("-v",),
            "word_exact": ("show", "get-url"),
        },
        "forbid_other_flags": True,
        "why": "`add` / `remove` / `rename` / `set-url` mutate.",
    },
    "stash": {
        "allow_bare": False,
        "safe_leading": {"word_exact": ("list", "show")},
        "forbid_other_flags": False,
        "why": "Bare `git stash` pushes; `pop` / `drop` / `apply` mutate.",
    },
    "clean": {
        "allow_bare": False,
        "safe_leading": {"flag_exact": ("-n", "--dry-run")},
        "forbid_other_flags": False,
        "why": "`-f` is destructive; only the dry-run form is safe.",
    },
}

# `-c` can override core.pager / diff.external / core.sshCommand, turning
# a "read" into arbitrary code execution. Strip these before exec.
ESCAPE_VAL_FLAGS = {
    "-c", "-C", "--config-env", "--exec-path",
    "--git-dir", "--work-tree", "--namespace",
}
ESCAPE_PREFIX_FLAGS = (
    "--config-env=", "--exec-path=", "--git-dir=",
    "--work-tree=", "--namespace=",
)


def _strip_escape_flags(argv: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in ESCAPE_VAL_FLAGS:
            i += 2
            continue
        if a.startswith(ESCAPE_PREFIX_FLAGS):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _leading_safe(spec: dict, argv: list[str]) -> bool:
    if not argv:
        return bool(spec.get("allow_bare"))
    sl = spec.get("safe_leading", {})
    head = argv[0]
    return (
        head in sl.get("flag_exact", ())
        or head in sl.get("word_exact", ())
        or any(head.startswith(p) for p in sl.get("flag_prefix", ()))
    )


def _validate_tier2(spec: dict, argv: list[str]) -> bool:
    if not _leading_safe(spec, argv):
        return False
    if argv and spec.get("forbid_other_flags"):
        return all(not a.startswith("-") for a in argv[1:])
    return True


_HOOK_TIER1_REDIRECT = frozenset({"status", "log", "diff", "show", "blame", "grep"})
_ADD_UPDATE_FORMS = frozenset({"-u", "--update"})


def _emit(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}))


def _hook_cli() -> None:
    seg = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        argv = shlex.split(seg)
    except ValueError:
        return
    if len(argv) < 2 or argv[0].rsplit("/", 1)[-1] != "git":
        return
    sub, rest = argv[1], argv[2:]

    if sub in _HOOK_TIER1_REDIRECT:
        _emit(f'Use mcp__dev-buddy__git_read(op="{sub}", args="...") '
              f'instead of git {sub}.')
        return

    if sub in TIER2 and _leading_safe(TIER2[sub], rest):
        _emit(f'Use mcp__dev-buddy__git_read(op="{sub}", args="...") '
              f'for this read-only form.')
        return

    if sub == "add":
        for tok in rest:
            if tok in _ADD_UPDATE_FORMS:
                _emit("Use mcp__dev-buddy__git_add_update instead of "
                      "git add --update.")
                return
        for tok in rest:
            if (
                tok.startswith("-")
                or tok in (".", "..", "./", "*")
                or any(ch in tok for ch in "*?[")
            ):
                continue
            r = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", tok],
                capture_output=True,
            )
            if r.returncode == 0:
                _emit(f"Use mcp__dev-buddy__git_add_update for "
                      f"tracked file '{tok}'.")
                return


if __name__ == "__main__":
    _hook_cli()
