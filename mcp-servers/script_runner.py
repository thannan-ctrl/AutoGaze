# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""script-runner MCP — launches tracked Python scripts in the background.

Bright-line safety: the script must be tracked in the current git repo
(`git ls-files --error-unmatch <script>` must succeed). Claude reads the
returned log_path with the built-in Read tool to tail output — no
separate job_status / tail_log tool needed.
"""
from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from _launch_helper import build_launch_cmd, validate_script
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("script-runner")

LOG_DIR = Path("/tmp/almost-autonomous-logs")


@mcp.tool()
def launch_script(script: str, args: str = "", log_path: str = "") -> dict:
    """Launch a git-tracked .py script in the background; returns {job_id, pid, log_path}.

    If log_path is set, write logs there (any absolute or relative path; parent
    dirs are created). Otherwise default to /tmp/almost-autonomous-logs/<job_id>.log.
    """
    try:
        validate_script(script)
    except ValueError as e:
        return {"error": str(e)}
    cmd = build_launch_cmd(script, args)
    job_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    if log_path:
        log_path = Path(log_path).expanduser().resolve()
    else:
        log_path = LOG_DIR / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    return {"job_id": job_id, "pid": proc.pid, "log_path": str(log_path)}


if __name__ == "__main__":
    mcp.run()
