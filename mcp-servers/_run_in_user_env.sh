#!/usr/bin/env sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run "$@" with the user's interactive shell env activated. Sources the
# non-login rc that matches $SHELL (.zshrc for zsh, .bashrc for bash) so
# conda activation, HF_TOKEN, etc. are picked up regardless of OS or
# login-shell chain. Missing rc files are silently skipped.
case "$(basename "${SHELL:-/bin/bash}")" in
  zsh)
    exec zsh  -c '[ -f "$HOME/.zshrc"  ] && . "$HOME/.zshrc";  exec "$@"' zsh  "$@" ;;
  *)
    exec bash -c '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"; exec "$@"' bash "$@" ;;
esac
