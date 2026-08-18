#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PreToolUse:Bash hook. Reads Claude Code's tool-input JSON from stdin,
# emits {"decision":"block","reason":"..."} to redirect Claude to the
# right tool, or exits silently to pass through.
#
# Tier 1 — universal Claude Code hygiene (always-on).
# Tier 2 — demo-specific MCP redirects. Add one Tier 2 rule per MCP tool you add.

cmd="$(jq -r '.tool_input.command // ""')"
WHITELIST_PY="$(cd "$(dirname "$0")/../mcp-servers" && pwd)/_git_whitelist.py"

block() {
  # Escape backslashes and double-quotes so the reason is valid JSON.
  local reason="$1"
  reason="${reason//\\/\\\\}"
  reason="${reason//\"/\\\"}"
  printf '{"decision":"block","reason":"%s"}\n' "$reason"
  exit 0
}

# Replace command-substitution wrappers with `;` separators so $(grep foo)
# and `grep foo` are surfaced as their own segments by the splitter below.
cmd_norm="$(printf '%s' "$cmd" | sed -E 's/\$\(/;/g; s/`/;/g; s/\)/;/g')"

# Quote-aware splitter: replace top-level &&, ||, ;, | with newlines while
# leaving separators inside "..." or '...' alone.
segments="$(printf '%s' "$cmd_norm" | awk '
{
  q = ""
  for (i = 1; i <= length($0); i++) {
    c = substr($0, i, 1); n = substr($0, i, 2)
    if (q == "") {
      if (c == "\"" || c == "'\''") { q = c; printf "%s", c; continue }
      if (n == "&&" || n == "||") { print ""; i++; continue }
      if (c == ";" || c == "|")    { print ""; continue }
    } else if (c == q) { q = "" }
    printf "%s", c
  }
  print ""
}')"

check_git() {
  local seg="$1"
  local out
  out="$(python3 "$WHITELIST_PY" "$seg" 2>/dev/null)"
  if [ -n "$out" ]; then
    printf '%s\n' "$out"
    exit 0
  fi
}

check_segment() {
  local seg="$1"
  # Strip leading whitespace.
  seg="${seg#"${seg%%[![:space:]]*}"}"
  # Strip leading `(` or `!` (subshell / negation).
  while :; do
    case "$seg" in
      '('*|'!'*)
        seg="${seg#?}"
        seg="${seg#"${seg%%[![:space:]]*}"}"
        ;;
      *) break ;;
    esac
  done
  # Strip a leading VAR=value assignment (FOO=1 grep -> grep).
  local first_word
  first_word="$(printf '%s' "$seg" | awk '{print $1}')"
  case "$first_word" in
    *=*)
      seg="$(printf '%s' "$seg" | awk '{$1=""; sub(/^ /,""); print}')"
      ;;
  esac

  local first second
  first="$(printf '%s' "$seg" | awk '{print $1}')"
  first="${first##*/}"   # /usr/bin/grep -> grep
  second="$(printf '%s' "$seg" | awk '{print $2}')"

  case "$first" in
    cat|head|tail|wc) block "Use the Read tool instead of $first." ;;
    sed|awk)          block "Use the Edit tool instead of $first." ;;
    find|ls)          block "Use the Glob tool instead of $first." ;;
    grep|rg)          block "Use the Grep tool instead of $first." ;;
    test|'['|'[[')
      printf '%s' "$seg" | grep -qE '(-[dfehLrswxOG])\b' \
        && block "Use the Glob tool instead of '$first -d/-f/-e' for file existence checks."
      ;;
    echo|printf)
      printf '%s' "$seg" | grep -qE '>\s*[^&]' \
        && block "Use the Write tool instead of shell redirection."
      ;;
    pytest)     block "Use mcp__dev-buddy__run_pytest instead of pytest." ;;
    pre-commit) block "Use mcp__dev-buddy__run_precommit instead of pre-commit." ;;
    date)       block "Use mcp__dev-buddy__now instead of date." ;;
    mkdir)      block "Use mcp__dev-buddy__create_dir instead of mkdir." ;;
    stat)       block "Use mcp__dev-buddy__stat (or Read/Glob for contents) instead of stat." ;;
    jq)         block "Use mcp__dev-buddy__jq instead of jq via Bash." ;;
    python|python3)
      printf '%s' "$seg" | grep -qE '\s-m\s+pytest(\s|$)' \
        && block "Use mcp__dev-buddy__run_pytest instead of python -m pytest."
      printf '%s' "$seg" | grep -qE '\s-c\s+.*\bjson\b' \
        && block "Use mcp__dev-buddy__jq (or Read on the file) instead of python -c for JSON parsing."
      ;;
    git) check_git "$seg" ;;
  esac
}

while IFS= read -r seg; do
  [ -z "$(printf '%s' "$seg" | tr -d '[:space:]')" ] && continue
  check_segment "$seg"
done <<EOF
$segments
EOF
exit 0
