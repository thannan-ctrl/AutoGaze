# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""slurm-buddy MCP -cluster reference, per-call SSH.

Requires SSH access to a Slurm login node. This is a *reference*; the
author's production `slurm` MCP is ~1000 LoC with batch/env/container/
results tooling. Every tool here spawns a fresh `ssh` -no persistent
tunnel, no port forwarding, no daemon. (That's FAQ Q1, shown in code.)

Config via `clusters.yaml` next to this file (`mcp-servers/clusters.yaml`). Schema:
    default: <cluster-name>      # optional; required only if >1 cluster
    clusters:
      <name>:
        host          - SSH hostname (matching ~/.ssh/config Host alias)
        account       - sbatch --account
        partition     - sbatch --partition
        home_dir      - remote workdir for scripts / wrappers / logs
        container     - pyxis/enroot image (required, srun --container-image)
        mounts        - bind-mounts string (required, srun --container-mounts)
        gpus_per_node - int, emitted as #SBATCH --gpus-per-node
        extra_sbatch  - optional list of cluster-default #SBATCH lines
        extra_srun    - optional list of cluster-default srun flags
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slurm-buddy")

_EXP_NAME_RE = re.compile(r"\A[A-Za-z0-9._-]+\Z")
_JOB_ID_RE = re.compile(r"\A\d+(_\d+)?\Z")

_HERE = Path(__file__).resolve().parent
_REPO_DIR = _HERE.parent
_CONFIG_PATH = _HERE / "clusters.yaml"


def _cfg(cluster: str = "") -> dict:
    config_path = _CONFIG_PATH
    if not config_path.is_file():
        raise RuntimeError(f"missing {config_path} - edit the committed template with your cluster details")
    data = yaml.safe_load(config_path.read_text()) or {}
    clusters = data.get("clusters") or {}
    if not clusters:
        raise RuntimeError(f"{config_path}: no clusters defined under `clusters:`")
    name = cluster or data.get("default") or (next(iter(clusters)) if len(clusters) == 1 else "")
    if not name:
        raise RuntimeError(
            f"multiple clusters defined ({', '.join(clusters)}); "
            "set `default:` in clusters.yaml or pass cluster=<name>"
        )
    if name not in clusters:
        raise RuntimeError(f"unknown cluster {name!r}; defined: {', '.join(clusters)}")
    entry = clusters[name] or {}
    required = ("host", "account", "partition", "home_dir", "container", "mounts", "gpus_per_node")
    missing = [k for k in required if entry.get(k) in (None, "")]
    if missing:
        raise RuntimeError(f"cluster {name!r}: missing required keys: {', '.join(missing)}")
    home_dir = entry["home_dir"]
    if "<user>" in home_dir:
        # Resolve <user> by aligning the template path against _REPO_DIR.
        # e.g. /home/scratch.<user>/agent/vlm-claude -> thannan_wwfo
        for tmpl_part, repo_part in zip(Path(home_dir).parts, _REPO_DIR.parts):
            if "<user>" in tmpl_part:
                prefix = tmpl_part.replace("<user>", "")
                home_dir = home_dir.replace("<user>", repo_part.replace(prefix, "", 1))
                break
    return {
        "name": name,
        "host": entry["host"],
        "account": entry["account"],
        "partition": entry["partition"],
        "home_dir": home_dir,
        "container": entry["container"],
        "mounts": entry["mounts"],
        "gpus_per_node": int(entry["gpus_per_node"]),
        "extra_sbatch": list(entry.get("extra_sbatch") or []),
        "extra_srun": list(entry.get("extra_srun") or []),
    }


def _ssh(host: str, *remote_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", host, *remote_args],
        capture_output=True, text=True,
    )


def _local_git(project_dir: str, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", project_dir, *args],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def _looks_like_script_path(token: str) -> bool:
    return "/" in token or token.endswith((".py", ".sh", ".bash"))


def _is_git_tracked(host: str, exp_path: str, token: str) -> bool:
    r = _ssh(
        host,
        f"git -C {shlex.quote(exp_path)} ls-files --error-unmatch {shlex.quote(token)}",
    )
    return r.returncode == 0


def _scp_write(host: str, content: str, remote_path: str) -> tuple[bool, str]:
    """Write `content` to `host:remote_path` via scp. Returns (ok, stderr_if_failed)."""
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / Path(remote_path).name
        local.write_text(content)
        r = subprocess.run(
            ["scp", str(local), f"{host}:{remote_path}"],
            capture_output=True, text=True,
        )
    return (r.returncode == 0, r.stderr)


def _render_sbatch(
    command: str, time: str, nodes: int, ntasks_per_node: int,
    sbatch_extra: list[str], srun_extra: list[str],
    cfg: dict, exp_path: str, job_dir: str,
) -> str:
    sbatch = [
        "#!/bin/bash",
        f"#SBATCH --account={cfg['account']}",
        f"#SBATCH --partition={cfg['partition']}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={ntasks_per_node}",
        f"#SBATCH --gpus-per-node={cfg['gpus_per_node']}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={job_dir}/slurm-%j.out",
        *(f"#SBATCH {e}" for e in cfg["extra_sbatch"]),
        *(f"#SBATCH {e}" for e in sbatch_extra),
    ]
    srun = [
        "srun \\",
        f"    --container-image={cfg['container']} \\",
        f"    --container-mounts={cfg['mounts']} \\",
        f"    --container-workdir={exp_path} \\",
        *(f"    {a} \\" for a in cfg["extra_srun"]),
        *(f"    {a} \\" for a in srun_extra),
        f"    bash -c {shlex.quote(command)}",
    ]
    return "\n".join([*sbatch, "set -euo pipefail", *srun, ""])


@mcp.tool()
def slurm_sinfo(cluster: str = "") -> list[dict]:
    """Return partition availability aggregated from sinfo.

    Each entry: {partition, available, idle_nodes, total_nodes, idle_cpus, total_cpus}.
    Sorted by idle_nodes descending. Use this to pick the best partition before submitting.
    """
    try:
        cfg = _cfg(cluster)
    except RuntimeError as e:
        return [{"error": str(e)}]

    r = _ssh(cfg["host"], "sinfo --format='%P %a %D %t %C' --noheader")
    if r.returncode != 0:
        return [{"error": r.stderr.strip()}]

    partitions: dict[str, dict] = defaultdict(lambda: {
        "available": False, "idle_nodes": 0, "mix_nodes": 0, "total_nodes": 0,
        "idle_cpus": 0, "total_cpus": 0,
    })
    for line in r.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        pname, avail, nodes_str, state, cpu_str = parts[0], parts[1], parts[2], parts[3], parts[4]
        pname = pname.rstrip("*$")
        try:
            nodes = int(nodes_str)
        except ValueError:
            continue
        try:
            cpu_parts = cpu_str.split("/")
            idle_cpus = int(cpu_parts[1])
            total_cpus = int(cpu_parts[3])
        except (IndexError, ValueError):
            idle_cpus = total_cpus = 0
        p = partitions[pname]
        if avail == "up":
            p["available"] = True
        p["total_nodes"] += nodes
        p["total_cpus"] += total_cpus
        state_clean = state.rstrip("*$-")
        if state_clean == "idle":
            p["idle_nodes"] += nodes
            p["idle_cpus"] += idle_cpus
        elif state_clean == "mix":
            p["mix_nodes"] += nodes

    result = [{"partition": name, **info} for name, info in partitions.items() if info["available"]]
    result.sort(key=lambda x: x["idle_nodes"] + x["mix_nodes"], reverse=True)
    return result


@mcp.tool()
def slurm_exp_init(exp_name: str, cluster: str = "") -> dict:
    """Clone origin@HEAD-branch into <home_dir>/<exp_name> on the cluster."""
    if not _EXP_NAME_RE.match(exp_name or ""):
        return {"error": "invalid exp_name", "detail": "must match [A-Za-z0-9._-]+"}
    try:
        cfg = _cfg(cluster)
        origin = _local_git(str(_REPO_DIR), "remote", "get-url", "origin")
        branch = _local_git(str(_REPO_DIR), "rev-parse", "--abbrev-ref", "HEAD")
    except RuntimeError as e:
        return {"error": str(e)}
    if branch == "HEAD":
        return {"error": "detached HEAD - check out a branch before slurm_exp_init"}

    remote_path = f"{cfg['home_dir']}/{exp_name}"
    test = _ssh(cfg["host"], f"test -e {shlex.quote(remote_path)} && echo EXISTS || echo MISSING")
    if test.returncode != 0:
        return {"error": "ssh probe failed", "detail": test.stderr or test.stdout}
    if "EXISTS" in test.stdout:
        return {"error": "remote path already exists", "remote_path": remote_path}

    if _ssh(cfg["host"], f"mkdir -p {shlex.quote(cfg['home_dir'])}").returncode != 0:
        return {"error": "mkdir failed"}

    clone = _ssh(
        cfg["host"],
        f"git clone --branch {shlex.quote(branch)} "
        f"{shlex.quote(origin)} {shlex.quote(remote_path)}",
    )
    if clone.returncode != 0:
        return {"error": "git clone failed", "detail": clone.stderr}
    return {"remote_path": remote_path, "branch": branch, "cluster": cfg["name"]}


@mcp.tool()
def slurm_submit(
    exp_name: str,
    command: str,
    time: str = "00:10:00",
    nodes: int = 1,
    ntasks_per_node: int = 1,
    sbatch_extra: list[str] = [],
    srun_extra: list[str] = [],
    cluster: str = "",
    container: str = "",
) -> dict:
    """Submit `command` to Slurm inside the cluster's container.

    Validates that any path-shaped token in `command` is git-tracked under
    `<home_dir>/<exp_name>` on the cluster. If validation fails, refuses to
    submit and returns the rendered wrapper plus a hint for a Bash fallback.

    container: optional override for the pyxis/enroot image; if omitted,
               uses the cluster-level `container` from clusters.yaml.
    """
    if not _EXP_NAME_RE.match(exp_name or ""):
        return {"error": "invalid exp_name", "detail": "must match [A-Za-z0-9._-]+"}
    try:
        cfg = _cfg(cluster)
    except RuntimeError as e:
        return {"error": str(e)}

    exp_path = f"{cfg['home_dir']}/{exp_name}"
    chk = _ssh(cfg["host"], f"test -d {shlex.quote(exp_path)} && echo OK || true")
    if "OK" not in chk.stdout:
        return {"error": "experiment folder missing", "detail": f"{exp_path} - run slurm_exp_init"}

    if container:
        cfg = {**cfg, "container": container}

    path_tokens = [t for t in shlex.split(command) if _looks_like_script_path(t)]
    prefix = Path(path_tokens[0]).stem if path_tokens else "job"
    job_dir = f"{exp_path}/.aa-jobs/{prefix}-{uuid.uuid4().hex[:6]}"

    wrapper_text = _render_sbatch(
        command, time, nodes, ntasks_per_node,
        sbatch_extra, srun_extra, cfg, exp_path, job_dir,
    )

    untracked = next(
        (t for t in path_tokens if not _is_git_tracked(cfg["host"], exp_path, t)),
        None,
    )
    if untracked is not None:
        return {
            "error": "untracked script",
            "detail": untracked,
            "rendered_sbatch": wrapper_text,
            "hint": (
                f"slurm-buddy refused because {untracked} isn't tracked at HEAD on "
                f"the cluster's clone. To submit anyway, write the rendered "
                f"sbatch to a tempfile, scp it to {exp_path}/<name>.sbatch, "
                "and run sbatch via Bash (which will prompt for approval)."
            ),
        }

    if _ssh(cfg["host"], f"mkdir -p {shlex.quote(job_dir)}").returncode != 0:
        return {"error": "mkdir failed"}

    wrapper_remote = f"{job_dir}/run.sbatch"
    ok, stderr = _scp_write(cfg["host"], wrapper_text, wrapper_remote)
    if not ok:
        return {"error": "scp wrapper failed", "detail": stderr}

    meta = {
        "command": command,
        "exp_name": exp_name,
        "time": time,
        "nodes": nodes,
        "ntasks_per_node": ntasks_per_node,
        "sbatch_extra": sbatch_extra,
        "srun_extra": srun_extra,
        "submitted_at": datetime.now(UTC).isoformat(),
        "cluster": cfg["name"],
    }

    sbatch = _ssh(cfg["host"], f"sbatch {shlex.quote(wrapper_remote)}")
    if sbatch.returncode != 0:
        return {"error": "sbatch failed", "detail": sbatch.stderr}
    match = re.search(r"Submitted batch job (\d+)", sbatch.stdout)
    if not match:
        return {"error": "could not parse job id", "stdout": sbatch.stdout}
    job_id = match.group(1)
    log_path = f"{job_dir}/slurm-{job_id}.out"

    meta["job_id"] = job_id
    meta["log_path"] = log_path
    meta_remote = f"{job_dir}/meta.json"
    ok, stderr = _scp_write(cfg["host"], json.dumps(meta, indent=2) + "\n", meta_remote)
    if not ok:
        return {"error": "scp meta failed", "detail": stderr}

    return {
        "job_id": job_id,
        "exp_name": exp_name,
        "job_dir": job_dir,
        "log_path": log_path,
        "wrapper": wrapper_remote,
        "meta_path": meta_remote,
        "rendered_sbatch": wrapper_text,
        "cluster": cfg["name"],
    }


@mcp.tool()
def slurm_status(job_id: str, cluster: str = "") -> str:
    """Return Slurm state for job_id (PENDING/RUNNING/COMPLETED/FAILED/...)."""
    if not _JOB_ID_RE.match(job_id or ""):
        return "error: invalid job_id"
    try:
        cfg = _cfg(cluster)
    except RuntimeError as e:
        return f"error: {e}"
    qjob = shlex.quote(job_id)
    r = _ssh(cfg["host"], f"squeue -j {qjob} -h -o %T")
    state = r.stdout.strip()
    if state:
        return state
    # Not in the queue - check sacct for terminal state.
    r2 = _ssh(cfg["host"], f"sacct -j {qjob} -X -n -o State")
    return r2.stdout.strip() or "UNKNOWN"


@mcp.tool()
def slurm_cancel(job_id: str, cluster: str = "") -> str:
    """Cancel Slurm job_id via scancel."""
    if not _JOB_ID_RE.match(job_id or ""):
        return "error: invalid job_id"
    try:
        cfg = _cfg(cluster)
    except RuntimeError as e:
        return f"error: {e}"
    r = _ssh(cfg["host"], f"scancel {shlex.quote(job_id)}")
    if r.returncode != 0:
        return f"error: {r.stderr.strip()}"
    return f"cancelled {job_id}"


@mcp.tool()
def slurm_read(path: str, tail: int = 200, head: int = 0, cluster: str = "") -> str:
    """tail/head/cat a remote file. Default: last 200 lines. Use absolute paths (~ not expanded)."""
    try:
        cfg = _cfg(cluster)
    except RuntimeError as e:
        return f"error: {e}"
    qpath = shlex.quote(path)
    if head > 0:
        cmd = f"head -n {int(head)} {qpath}"
    elif tail > 0:
        cmd = f"tail -n {int(tail)} {qpath}"
    else:
        cmd = f"cat {qpath}"
    r = _ssh(cfg["host"], cmd)
    if r.returncode != 0:
        return f"error: {r.stderr.strip()}"
    return r.stdout


@mcp.tool()
def slurm_ls(path: str, recursive: bool = False, glob: str = "", cluster: str = "") -> str:
    """ls -la (default), or `find` if recursive=True or glob is set. Use absolute paths."""
    try:
        cfg = _cfg(cluster)
    except RuntimeError as e:
        return f"error: {e}"
    qpath = shlex.quote(path)
    if glob:
        cmd = f"find {qpath} -name {shlex.quote(glob)}"
    elif recursive:
        cmd = f"find {qpath}"
    else:
        cmd = f"ls -la {qpath}"
    r = _ssh(cfg["host"], cmd)
    if r.returncode != 0:
        return f"error: {r.stderr.strip()}"
    return r.stdout


@mcp.tool()
def slurm_grep(
    pattern: str, path: str,
    ignore_case: bool = False, max_matches: int = 200, cluster: str = "",
) -> str:
    """Recursive grep -n. Bounded by max_matches lines of output. Use absolute paths."""
    try:
        cfg = _cfg(cluster)
    except RuntimeError as e:
        return f"error: {e}"
    flags = "-rn" + ("i" if ignore_case else "")
    cmd = (
        f"grep {flags} -- {shlex.quote(pattern)} {shlex.quote(path)} "
        f"| head -n {int(max_matches)}"
    )
    r = _ssh(cfg["host"], cmd)
    # grep exits 1 on no-match; treat that as empty result, not error.
    if r.returncode not in (0, 1):
        return f"error: {r.stderr.strip()}"
    return r.stdout


@mcp.tool()
def slurm_select_partition(qformat: str = "", cluster: str = "") -> dict:
    """Pick the best partition for qformat from partition_candidates and update clusters.yaml.

    Reads partition_candidates[qformat] from clusters.yaml, queries sinfo for
    idle+mix node counts, selects the candidate with the most available nodes,
    and writes the result back to clusters.yaml. Returns the selected partition
    and node counts for all candidates.

    If qformat is omitted, considers all candidates across all qformats.
    Call this once at the start of each pipeline run (before slurm_submit).
    """
    data = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    clusters_cfg = data.get("clusters") or {}
    cluster_name = cluster or data.get("default") or (
        next(iter(clusters_cfg)) if len(clusters_cfg) == 1 else ""
    )
    if not cluster_name or cluster_name not in clusters_cfg:
        return {"error": f"unknown cluster {cluster_name!r}"}

    entry = clusters_cfg[cluster_name]
    candidates_map: dict = entry.get("partition_candidates") or {}
    if not candidates_map:
        return {"error": "no partition_candidates defined in clusters.yaml"}

    if qformat:
        if qformat not in candidates_map:
            return {"error": f"no partition_candidates for qformat {qformat!r}"}
        candidate_set = set(candidates_map[qformat])
    else:
        candidate_set = {p for ps in candidates_map.values() for p in ps}

    try:
        cfg = _cfg(cluster)
    except RuntimeError as e:
        return {"error": str(e)}

    r = _ssh(cfg["host"], "sinfo --format='%P %a %D %t' --noheader")
    if r.returncode != 0:
        return {"error": r.stderr.strip()}

    counts: dict[str, dict] = defaultdict(lambda: {"idle_nodes": 0, "mix_nodes": 0, "total_nodes": 0})
    for line in r.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        pname, avail, nodes_str, state = parts[0], parts[1], parts[2], parts[3]
        pname = pname.rstrip("*$")
        if pname not in candidate_set or avail != "up":
            continue
        try:
            nodes = int(nodes_str)
        except ValueError:
            continue
        p = counts[pname]
        p["total_nodes"] += nodes
        state_clean = state.rstrip("*$-")
        if state_clean == "idle":
            p["idle_nodes"] += nodes
        elif state_clean == "mix":
            p["mix_nodes"] += nodes

    scored = [
        {
            "partition": p,
            "idle_nodes": counts[p]["idle_nodes"],
            "mix_nodes": counts[p]["mix_nodes"],
            "total_nodes": counts[p]["total_nodes"],
            "available_nodes": counts[p]["idle_nodes"] + counts[p]["mix_nodes"],
        }
        for p in candidate_set
    ]
    scored.sort(key=lambda x: x["available_nodes"], reverse=True)
    best = scored[0]["partition"]

    entry["partition"] = best
    _CONFIG_PATH.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    return {
        "selected": best,
        "qformat": qformat or "(all)",
        "candidates": scored,
        "updated": str(_CONFIG_PATH),
    }


if __name__ == "__main__":
    mcp.run()
