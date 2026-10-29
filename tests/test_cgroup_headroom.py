"""Tests for src.paco_base.cgroup_memory_headroom on fake cgroup trees.

Run: python -m pytest tests/  (or: python tests/test_cgroup_headroom.py)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paco_base import cgroup_memory_headroom

GiB = 1024 ** 3


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _fake(cgroup_line):
    root = tempfile.mkdtemp()
    proc, cg = os.path.join(root, "proc"), os.path.join(root, "cgroup")
    _write(os.path.join(proc, "self", "cgroup"), cgroup_line + "\n")
    return proc, cg


def test_v2_limit_on_ancestor():
    # SLURM-like: limit on the job, process in a task cgroup below it.
    proc, cg = _fake("0::/slurm/job_42/step_0/task_0")
    _write(f"{cg}/slurm/job_42/step_0/task_0/memory.max", "max\n")
    _write(f"{cg}/slurm/job_42/step_0/task_0/memory.current", f"{1 * GiB}\n")
    _write(f"{cg}/slurm/job_42/memory.max", f"{8 * GiB}\n")
    _write(f"{cg}/slurm/job_42/memory.current", f"{3 * GiB}\n")
    assert cgroup_memory_headroom(proc, cg) == 5 * GiB


def test_v2_smallest_headroom_wins():
    proc, cg = _fake("0::/a/b")
    _write(f"{cg}/a/b/memory.max", f"{4 * GiB}\n")
    _write(f"{cg}/a/b/memory.current", f"{1 * GiB}\n")   # 3 GiB left
    _write(f"{cg}/a/memory.max", f"{10 * GiB}\n")
    _write(f"{cg}/a/memory.current", f"{9 * GiB}\n")     # 1 GiB left
    assert cgroup_memory_headroom(proc, cg) == 1 * GiB


def test_v2_unlimited():
    proc, cg = _fake("0::/user.slice")
    _write(f"{cg}/user.slice/memory.max", "max\n")
    _write(f"{cg}/user.slice/memory.current", f"{1 * GiB}\n")
    assert cgroup_memory_headroom(proc, cg) is None


def test_v1_limit():
    proc, cg = _fake("4:memory:/docker/abc")
    _write(f"{cg}/memory/docker/abc/memory.limit_in_bytes", f"{2 * GiB}\n")
    _write(f"{cg}/memory/docker/abc/memory.usage_in_bytes", f"{GiB // 2}\n")
    assert cgroup_memory_headroom(proc, cg) == 2 * GiB - GiB // 2


def test_v1_unlimited_sentinel():
    proc, cg = _fake("4:memory:/")
    _write(f"{cg}/memory/memory.limit_in_bytes", "9223372036854771712\n")
    _write(f"{cg}/memory/memory.usage_in_bytes", f"{GiB}\n")
    assert cgroup_memory_headroom(proc, cg) is None


def test_usage_above_limit_gives_zero():
    proc, cg = _fake("0::/job")
    _write(f"{cg}/job/memory.max", f"{1 * GiB}\n")
    _write(f"{cg}/job/memory.current", f"{2 * GiB}\n")
    assert cgroup_memory_headroom(proc, cg) == 0


def test_no_cgroups():
    assert cgroup_memory_headroom("/nonexistent-proc", "/nonexistent-cg") is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
