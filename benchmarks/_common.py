"""
Shared helpers for the PACO parallelization benchmarks in this directory.

These scripts are meant to be portable: run them here, or copy the whole
`benchmarks/` + `src/` (+ optionally `challenge/subchallenge_1/*.fits` for
real data) tree to a different machine (e.g. a compute server) and run
there. All paths are resolved relative to this file, not hardcoded.
"""
import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Repo root = parent of this file's directory (benchmarks/../)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil
import threading


class PeakMemoryMonitor:
    """
    Context manager that samples RSS of the current process + all of its
    (recursive) child processes in a background thread, tracking:
      - peak_parent_mb: peak RSS of the main process alone.
      - peak_single_child_mb: peak RSS of any *one* worker process.
      - peak_combined_mb: peak of (parent RSS + sum of children RSS) at a
        single sampling instant.

    Caveat (important for comparing a shared-memory design against one that
    isn't): RSS counts every page a process has *mapped and touched*,
    including pages that are physically shared with other processes (e.g.
    multiprocessing.shared_memory segments, or copy-on-write pages nobody
    has written to yet). `peak_combined_mb` sums RSS across processes, so it
    double-counts truly shared physical memory as if each process held a
    private copy — this makes it a conservative *upper bound*, and if
    anything unfairly penalises the shared-memory design more than the one
    that privately copies data to each worker. Getting an exact
    de-duplicated figure (PSS) would require reading /proc/*/smaps, which
    doesn't exist on macOS — not attempted here for portability. Use
    peak_parent_mb + peak_single_child_mb for a fairer *marginal* estimate
    of what one extra worker actually costs on top of the shared baseline.

    Usage:
        with PeakMemoryMonitor() as mon:
            ... do work (possibly spawning subprocesses) ...
        print(mon.peak_combined_mb)
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_parent_mb = 0.0
        self.peak_single_child_mb = 0.0
        self.peak_combined_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process()

    def _sample_once(self):
        try:
            parent_rss = self._proc.memory_info().rss
        except Exception:
            return
        children_rss = []
        try:
            for child in self._proc.children(recursive=True):
                try:
                    children_rss.append(child.memory_info().rss)
                except Exception:
                    pass  # child may have just exited
        except Exception:
            pass
        combined = parent_rss + sum(children_rss)
        max_child = max(children_rss) if children_rss else 0

        self.peak_parent_mb = max(self.peak_parent_mb, parent_rss / 1024**2)
        self.peak_single_child_mb = max(self.peak_single_child_mb, max_child / 1024**2)
        self.peak_combined_mb = max(self.peak_combined_mb, combined / 1024**2)

    def _run(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1)
        self._sample_once()  # catch a final peak right at teardown
        return False


class CgroupMemoryMonitor:
    """
    Linux-only. Measures the REAL peak physical memory of the current
    process tree via the kernel's own cgroup v2 accounting, instead of
    psutil-based RSS polling (see PeakMemoryMonitor's docstring for why
    summing RSS across processes over-counts physically shared pages).
    cgroup accounting is exact and naturally de-duplicates physically
    shared pages (e.g. our own multiprocessing.shared_memory segments, or
    copy-on-write pages): a shared page is charged once to whichever
    cgroup first touched it, not once per process that maps it. This is
    the single, defensible "real memory used" number PeakMemoryMonitor's
    docstring says psutil cannot give on macOS (no /proc/*/smaps there);
    on Linux, this is that number.

    How it works: creates a fresh leaf cgroup under this process's own
    current cgroup and moves the current process into it. Multiprocessing
    workers forked afterward automatically inherit that cgroup membership
    (cgroup v2: a forked child stays in its parent's cgroup unless
    explicitly moved), so PACOCalc's whole worker pool is covered without
    any per-worker bookkeeping. Moves the process back to the original
    cgroup and removes the leaf on exit.

    Peak tracking is a HYBRID of two sources, not a single end-of-run
    read of memory.peak -- confirmed necessary empirically, not just
    theoretically: on a real server run under a systemd-managed,
    non-delegated scope (e.g. a tmux pane's own transient scope), the
    leaf cgroup directory was observed to disappear (ENOENT) by the time
    __exit__ tried to read memory.peak, even though this process was
    still alive and presumably still resident in it moments before --
    consistent with systemd (or an equivalent cgroup-empty reclaim path)
    periodically pruning sub-cgroups it does not itself manage, which can
    race an external, unprivileged script's own leaf regardless of
    whether our own PID is technically still in cgroup.procs at read
    time. Relying solely on a point-in-time read at teardown is therefore
    NOT robust on every server configuration, however correct it is in
    principle. The fix: a background thread polls this leaf's
    memory.current (the live usage counter, distinct from memory.peak's
    historical-max counter) at a short interval and keeps a running max
    in self.peak_mb, exactly like PeakMemoryMonitor's own psutil polling
    -- so even if the leaf vanishes right at teardown, whatever was
    already sampled survives. memory.peak is still read once at the end
    as a bonus and folded in via max() when available, since when it DOES
    survive to teardown it is strictly more precise than polling (kernel
    event-driven, not subject to this thread's own polling-interval blind
    spot) -- this hybrid gets the best available number in either case
    rather than choosing one approach and accepting that approach's own
    single failure mode.

    Preconditions (checked at construction; raises RuntimeError with a
    specific, actionable message if unmet -- see make_memory_monitor()
    for a wrapper that catches this and falls back to PeakMemoryMonitor):
      - Linux, with cgroup v2 (the unified hierarchy) mounted at
        /sys/fs/cgroup -- true by default on any reasonably recent distro.
      - The memory controller enabled for this process's current cgroup
        (typically true in a systemd user/system session or inside a
        container that owns its own cgroup; check
        /sys/fs/cgroup/<path>/cgroup.controllers if unsure).
      - Write permission to create a subdirectory there (true for a
        delegated user slice, a container's own cgroup, or root; if not,
        run the whole script under
        `systemd-run --scope -p MemoryAccounting=yes python3 ...`, which
        grants exactly this delegation for the scope's lifetime -- also
        the fix if you see this class's leaf-cgroup-vanished warning
        under a non-delegated scope, since Delegate=yes is what tells
        systemd not to prune the subtree it no longer directly manages).
      - Write permission on the parent's cgroup.subtree_control, so the
        memory controller can be propagated to the leaf this class
        creates (attempted automatically in __enter__, right after
        moving this process into the leaf -- see that method's
        comments). Without it, the leaf exists and can hold processes
        but has no memory.current/memory.peak files at all, which looks
        like "could not read memory.peak OR ever sample memory.current"
        (peak_mb stuck at 0.0) rather than the leaf-vanished warning
        above. Seen even under `systemd-run --scope -p Delegate=yes
        -p MemoryAccounting=yes`: MemoryAccounting=yes enables the
        controller for the scope's own cgroup but does not itself write
        "+memory" into that scope's subtree_control.

    Usage:
        with CgroupMemoryMonitor() as mon:
            ... do work (possibly spawning subprocesses via fork) ...
        print(mon.peak_mb)   # single real number, MB
    """

    def __init__(self, interval: float = 0.05):
        if sys.platform != "linux":
            raise RuntimeError(
                "CgroupMemoryMonitor requires Linux (cgroup v2); running "
                f"on {sys.platform!r}. Use PeakMemoryMonitor instead.")
        self._own_cgroup = self._read_own_cgroup()
        if self._own_cgroup is None:
            raise RuntimeError(
                "Could not determine this process's cgroup v2 path from "
                "/proc/self/cgroup -- is cgroup v2 (the unified hierarchy) "
                "mounted and in use on this system? (hybrid/legacy cgroup "
                "v1 setups are not supported here.)")
        base = os.path.join("/sys/fs/cgroup", self._own_cgroup.lstrip("/"))
        controllers_path = os.path.join(base, "cgroup.controllers")
        try:
            with open(controllers_path) as f:
                controllers = f.read().split()
        except OSError as exc:
            raise RuntimeError(
                f"Could not read {controllers_path}: {exc}. Is cgroup v2 "
                "mounted at /sys/fs/cgroup?") from exc
        if "memory" not in controllers:
            raise RuntimeError(
                f"The memory controller is not available in this cgroup "
                f"({controllers_path} lists: {controllers or '(none)'}). "
                "Ask your sysadmin to delegate it, or run this whole "
                "script under 'systemd-run --scope "
                "-p MemoryAccounting=yes python3 ...'.")
        self._base_dir = base
        self._leaf_dir = None
        self.interval = interval
        self.peak_mb = 0.0
        self._n_samples = 0
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _read_own_cgroup():
        try:
            with open("/proc/self/cgroup") as f:
                for line in f:
                    # cgroup v2 (unified hierarchy): a single line "0::/path".
                    parts = line.strip().split(":", 2)
                    if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
                        return parts[2]
        except OSError:
            pass
        return None

    def _sample_once(self):
        try:
            with open(os.path.join(self._leaf_dir, "memory.current")) as f:
                current = int(f.read().strip())
        except (OSError, ValueError):
            return  # leaf momentarily unreadable/gone -- skip, don't crash a bg thread
        self.peak_mb = max(self.peak_mb, current / 1024**2)
        self._n_samples += 1

    def _run(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def __enter__(self):
        leaf = os.path.join(self._base_dir, f"paco_bench_{os.getpid()}_{id(self)}")
        try:
            os.makedirs(leaf, exist_ok=False)
            with open(os.path.join(leaf, "cgroup.procs"), "w") as f:
                f.write(str(os.getpid()))
        except OSError as exc:
            raise RuntimeError(
                f"Could not create/join leaf cgroup {leaf}: {exc}. Need "
                "write permission on the parent cgroup (delegated user "
                "slice, container cgroup, or root).") from exc
        self._leaf_dir = leaf
        # memory.current/memory.peak only exist in a cgroup once the
        # memory controller is actually enabled THERE, which for a
        # freshly created child is controlled by the PARENT's
        # cgroup.subtree_control -- NOT the parent's own
        # cgroup.controllers (already checked in __init__: that only
        # says the controller is available/enabled for the parent's OWN
        # accounting, it says nothing about propagation to children).
        # Observed on a systemd-run --scope -p Delegate=yes
        # -p MemoryAccounting=yes scope: MemoryAccounting=yes enables
        # "memory" for the scope's own cgroup but does not write
        # "+memory" into that scope's subtree_control, so this leaf had
        # no memory.current/memory.peak files at all -- every read was
        # ENOENT, from the very first sample onward (not just at
        # teardown like the leaf-vanishing case this class's docstring
        # already covers). Enable it ourselves here -- but only AFTER
        # moving our own process into the leaf above: cgroup v2's "no
        # internal process constraint" refuses to enable a controller in
        # a cgroup's subtree_control while that cgroup still has a
        # member process directly, which was still true before the
        # migration just above.
        try:
            with open(os.path.join(self._base_dir, "cgroup.subtree_control")) as f:
                enabled = f.read().split()
            if "memory" not in enabled:
                with open(os.path.join(self._base_dir, "cgroup.subtree_control"), "w") as f:
                    f.write("+memory")
        except OSError:
            pass  # best-effort; _sample_once()'s own try/except covers the fallout
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1)
        self._sample_once()  # catch a final memory.current peak right at teardown

        peak_read_ok = False
        try:
            with open(os.path.join(self._leaf_dir, "memory.peak")) as f:
                peak_value_mb = int(f.read().strip()) / 1024**2
            self.peak_mb = max(self.peak_mb, peak_value_mb)
            peak_read_ok = True
        except OSError as exc:
            if self._n_samples > 0:
                print(f"[CgroupMemoryMonitor] memory.peak unavailable at "
                     f"teardown ({exc}); using the background-sampled "
                     f"peak from {self._n_samples} memory.current reads "
                     f"instead ({self.peak_mb:.1f} MB) -- see this class's "
                     f"docstring on why the leaf cgroup can vanish before "
                     f"teardown under a non-delegated systemd scope.",
                     file=sys.stderr)
            else:
                print(f"[CgroupMemoryMonitor] could not read memory.peak "
                     f"OR ever sample memory.current from {self._leaf_dir} "
                     f"({exc}) -- peak_mb is unreliable (0.0).",
                     file=sys.stderr)

        # Move back to the parent cgroup before removing the leaf -- rmdir
        # fails while any PID (i.e. this process) remains in it. Wrapped so
        # a failure here (e.g. the leaf already gone) never discards the
        # peak_mb value already captured above.
        try:
            with open(os.path.join(self._base_dir, "cgroup.procs"), "w") as f:
                f.write(str(os.getpid()))
        except OSError:
            pass
        try:
            os.rmdir(self._leaf_dir)
        except OSError:
            pass
        return False


class PssMemoryMonitor:
    """
    Linux-only. Like PeakMemoryMonitor (samples this process + all
    recursive children in a background thread), but sums each process's
    Pss (proportional set size, from /proc/<pid>/smaps_rollup) instead of
    RSS -- which fixes the exact double-counting problem PeakMemoryMonitor's
    own docstring warns about, WITHOUT any of CgroupMemoryMonitor's
    machinery: a page shared between N of our own processes (e.g. a
    multiprocessing.shared_memory segment, or a copy-on-write page nobody
    has written to yet) counts towards each mapper's Pss divided by N, so
    summing Pss across our whole process tree gives a number close to the
    actual unique physical memory that tree holds, not N times it. This is
    the same technique the standard `smem` tool uses, applied to just our
    own process tree.

    Preferred over CgroupMemoryMonitor on a server where cgroup v2
    delegation is awkward: needs no root, no systemd delegation, and no
    special launch command (no `systemd-run --scope -p Delegate=yes
    -p MemoryAccounting=yes ...` wrapper) -- /proc/<pid>/smaps_rollup is
    readable for any process you own on any mainline kernel >= 4.14.
    CgroupMemoryMonitor's leaf-cgroup-creation dance turned out to have
    several sharp, hard-to-fully-characterize edges on a non-trivially
    configured systemd/cgroup-v2 host (see its own docstring and this
    module's git history for the specific failure modes hit in practice:
    the leaf vanishing before teardown, the memory controller never being
    propagated to a freshly created child at all, and -- after trying to
    fix that by enabling it ourselves -- the process getting stuck unable
    to migrate back out of the leaf, nesting a new leaf under the old one
    on every subsequent run). None of that applies here: this class never
    creates a cgroup, writes to cgroup.procs, or touches
    cgroup.subtree_control -- it only ever reads files that already exist
    for any process.

    Caveat, for completeness: if one of our pages is ALSO mapped by some
    unrelated process outside our own tree (e.g. a shared libc .so page
    mapped by many other, unrelated processes on the system), Pss divides
    its cost by ALL mappers system-wide, not just the ones in our tree --
    so this slightly UNDER-counts what our tree would actually free if it
    exited, in contrast to RSS-summing's much larger OVER-count in the
    other direction. This is the standard, well-understood tradeoff Pss
    accounting makes (same one `smem` accepts), and in practice the error
    it introduces is small and confined to shared libraries, not our own
    cube/patch data (which nothing outside our own process tree maps).

    Usage:
        with PssMemoryMonitor() as mon:
            ... do work (possibly spawning subprocesses) ...
        print(mon.peak_mb)
    """

    def __init__(self, interval: float = 0.05):
        if sys.platform != "linux":
            raise RuntimeError(
                "PssMemoryMonitor requires Linux (/proc/<pid>/smaps_rollup); "
                f"running on {sys.platform!r}. Use PeakMemoryMonitor instead.")
        try:
            with open("/proc/self/smaps_rollup") as f:
                if "Pss:" not in f.read():
                    raise RuntimeError(
                        "/proc/self/smaps_rollup exists but has no 'Pss:' "
                        "line -- is CONFIG_PROC_PAGE_MONITOR enabled in "
                        "this kernel?")
        except OSError as exc:
            raise RuntimeError(
                f"Could not read /proc/self/smaps_rollup: {exc}.") from exc
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process()

    @staticmethod
    def _pss_kb(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/smaps_rollup") as f:
                for line in f:
                    if line.startswith("Pss:"):
                        return int(line.split()[1])  # kB, per the kernel's own units
        except (OSError, ValueError, IndexError):
            pass  # process may have just exited, or be momentarily unreadable
        return 0

    def _sample_once(self):
        try:
            pids = [self._proc.pid] + [c.pid for c in self._proc.children(recursive=True)]
        except Exception:
            return
        total_kb = sum(self._pss_kb(pid) for pid in pids)
        if total_kb > 0:
            self.peak_mb = max(self.peak_mb, total_kb / 1024)

    def _run(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1)
        self._sample_once()
        return False


def make_memory_monitor(args):
    """
    Return a memory-monitor context manager per args.memory_backend (see
    make_arg_parser's --memory-backend help for the tradeoffs):
      - "psutil" (default): PeakMemoryMonitor -- portable, but an
        estimate that double-counts physically shared pages.
      - "pss": PssMemoryMonitor -- Linux only, sums de-duplicated Pss
        across the process tree from /proc/<pid>/smaps_rollup. The
        recommended real-memory backend on a server: no systemd/cgroup
        setup needed, no fragile delegation to get wrong (see that
        class's docstring for why "cgroup" below ended up unreliable in
        practice).
      - "cgroup": CgroupMemoryMonitor -- Linux only, a single real number
        from the kernel's own cgroup v2 accounting. Exact in principle,
        but depends on this cgroup's delegation setup being exactly
        right (see that class's docstring); prefer "pss" unless you
        specifically need this. Falls back to PeakMemoryMonitor with a
        printed warning if this host doesn't meet its preconditions.
    """
    backend = getattr(args, "memory_backend", "psutil")
    if backend == "pss":
        try:
            return PssMemoryMonitor()
        except RuntimeError as exc:
            print(f"[memory-backend] pss unavailable ({exc}); falling "
                 "back to psutil-based PeakMemoryMonitor.", file=sys.stderr)
    elif backend == "cgroup":
        try:
            return CgroupMemoryMonitor()
        except RuntimeError as exc:
            print(f"[memory-backend] cgroup unavailable ({exc}); falling "
                 "back to psutil-based PeakMemoryMonitor.", file=sys.stderr)
    return PeakMemoryMonitor()


def mem_columns(mon) -> dict:
    """
    Uniform dict of table/CSV memory columns for any monitor type, so
    callers don't need an isinstance check at every call site. Exactly one
    of ("mem_parent_mb", "mem_maxworker_mb", "mem_combined_mb") vs.
    ("mem_real_mb") is populated; the others are left blank rather than 0,
    so an empty cell in a CSV clearly means "not measured this way" rather
    than "measured as zero". mem_real_mb holds whichever single-number
    real-memory backend actually ran (pss or cgroup) -- they're mutually
    exclusive per run, so one shared column is enough; the console summary
    (mem_str) still says which one it was.
    """
    if isinstance(mon, (PssMemoryMonitor, CgroupMemoryMonitor)):
        return {"mem_parent_mb": "", "mem_maxworker_mb": "",
                "mem_combined_mb": "", "mem_real_mb": f"{mon.peak_mb:.0f}"}
    return {"mem_parent_mb": f"{mon.peak_parent_mb:.0f}",
            "mem_maxworker_mb": f"{mon.peak_single_child_mb:.0f}",
            "mem_combined_mb": f"{mon.peak_combined_mb:.0f}",
            "mem_real_mb": ""}


def mem_str(mon) -> str:
    """Uniform one-line printable memory summary for any monitor type."""
    if isinstance(mon, PssMemoryMonitor):
        return f"real(pss)={mon.peak_mb:.0f}MB"
    if isinstance(mon, CgroupMemoryMonitor):
        return f"real(cgroup)={mon.peak_mb:.0f}MB"
    return (f"parent={mon.peak_parent_mb:.0f}MB "
           f"maxWorker={mon.peak_single_child_mb:.0f}MB "
           f"combined(upper-bound)={mon.peak_combined_mb:.0f}MB")


def default_data_dir() -> Path:
    return REPO_ROOT / "challenge" / "subchallenge_1"


def load_real_dataset(data_dir: Path, name: str = "sphere_irdis", idx: int = 3):
    """Load a real challenge dataset (cube, angles, psf, fwhm) if present."""
    from astropy.io import fits

    def fp(kind):
        return data_dir / f"{name}_{kind}_{idx}.fits"

    cube = fits.getdata(fp("cube")).astype(np.float64)
    pa = fits.getdata(fp("pa")).astype(np.float64).ravel()
    psf = fits.getdata(fp("psf")).astype(np.float64)
    if psf.ndim == 3:
        psf = np.nanmedian(psf, axis=0)
    return cube, pa, psf


def load_epoch_dataset(data_dir: Path):
    """
    Load a real, pre-processing "epoch" cube (secondepoch/, firstepoch/):
    center_im-2.fits (cube), rotnth.fits (parallactic angles), finalPSF.fits
    (2-D PSF). Different naming convention and different instrument than
    the sph3 challenge data (load_real_dataset) — used here purely as a
    large, real full-size stress-test cube, not for detection validation.

    Verified directly against the actual FITS headers (astropy already
    returns cube axes as (T, H, W) — T first, matching PACO's convention;
    no transpose needed): secondepoch/center_im-2.fits -> (368, 1280, 2048),
    rotnth.fits -> (368,) matching cube.shape[0] exactly. cube.shape[0] is
    always T for these files, regardless of on-disk FITS NAXIS order — do
    not assume this holds for a cube from a different pipeline without
    checking, since NAXIS order conventions vary by instrument.

    The raw cube is large (secondepoch: ~7.7 GB as float64) and loaded
    fully into RAM (not memmapped) so a benchmark's timing reflects
    compute/cache behaviour rather than lazy-load disk I/O — make sure the
    machine has enough RAM before using this on the "secondepoch" cube.
    """
    from astropy.io import fits

    cube = fits.getdata(data_dir / "center_im-2.fits").astype(np.float64)
    pa = fits.getdata(data_dir / "rotnth.fits").astype(np.float64).ravel()
    psf = fits.getdata(data_dir / "finalPSF.fits").astype(np.float64)
    if psf.ndim == 3:
        psf = np.nanmedian(psf, axis=0)
    return cube, pa, psf


def make_synthetic_dataset(T: int, H: int, W: int, fwhm: float = 4.0, seed: int = 0):
    """
    Generate a synthetic ADI-like cube for scalability testing at sizes real
    challenge data doesn't cover on this machine (e.g. LMIRCam-scale T).

    Not astrophysically realistic (just correlated-ish noise + a smooth PSF)
    — only meant to exercise the parallelization code paths at a chosen
    (T, H, W, K) size, not to validate detection performance.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, size=(H, W)).astype(np.float64)
    cube = np.stack([
        base + rng.normal(0, 0.3, size=(H, W)) for _ in range(T)
    ]).astype(np.float64)
    pa = np.linspace(0, 60, T)

    psf_size = max(15, int(6 * fwhm) | 1)  # odd size
    yy, xx = np.mgrid[0:psf_size, 0:psf_size]
    cy = cx = psf_size // 2
    sigma = fwhm / 2.355
    psf = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return cube, pa, psf


def parse_nproc_list(arg: str, cpu_count: int) -> list:
    """
    Parse a --nproc argument. Accepts:
      "auto"        -> [1, 2, 4, 8, 16, ...] up to cpu_count, plus cpu_count
      "2,4,8"       -> explicit comma list
    """
    if arg == "auto":
        vals = []
        n = 1
        while n < cpu_count:
            vals.append(n)
            n *= 2
        if cpu_count not in vals:
            vals.append(cpu_count)
        return sorted(set(vals))
    return sorted(set(int(x) for x in arg.split(",")))


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--nproc", default="auto",
                   help="Comma list (e.g. '2,4,8') or 'auto' to sweep "
                        "powers of 2 up to os.cpu_count() (default: auto).")
    p.add_argument("--data", choices=["real", "synthetic", "epoch"], default="real",
                   help="Use real challenge data (default, sph3), a "
                        "synthetic cube sized via --shape, or a real "
                        "pre-processing 'epoch' cube (secondepoch/ by "
                        "default — a different, larger instrument cube, "
                        "used as a real-data stress test rather than for "
                        "detection validation; see --data-dir to pick "
                        "firstepoch/ instead).")
    p.add_argument("--data-dir", default=None,
                   help="Override path to the dataset directory (default: "
                        "<repo>/challenge/subchallenge_1 for --data real, "
                        "<repo>/secondepoch for --data epoch).")
    p.add_argument("--shape", default="228,160,160",
                   help="T,H,W for --data synthetic (default: 228,160,160, "
                        "sph3-sized). Try e.g. 4800,200,200 for an "
                        "LMIRCam-scale scalability test.")
    p.add_argument("--fwhm", type=float, default=3.72,
                   help="FWHM in pixels (default: 3.72, sph3's value).")
    p.add_argument("--npix", type=int, default=None,
                   help="Number of test positions. Default: a size "
                        "appropriate to the benchmark. Ignored if "
                        "--sampling full.")
    p.add_argument("--sampling", choices=["dense", "random", "full"],
                   default="dense",
                   help="'dense' (default): a contiguous block, "
                        "representative of a real sweep's local density "
                        "per radius ring. 'random': scattered subset (NOT "
                        "representative of the ring-cache benefit, but "
                        "cheap for correctness checks across varied radii). "
                        "'full': every native-grid pixel.")
    p.add_argument("--csv", default=None,
                   help="Optional path to write results as CSV.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--memory-backend", choices=["psutil", "pss", "cgroup"],
                   default="psutil",
                   help="How to measure peak memory (default: psutil). "
                        "'psutil': portable RSS polling across the "
                        "process tree -- an upper-bound ESTIMATE that "
                        "double-counts physically shared pages (see "
                        "PeakMemoryMonitor's docstring). 'pss' "
                        "(RECOMMENDED on a Linux server): sums "
                        "de-duplicated Pss across the process tree from "
                        "/proc/<pid>/smaps_rollup -- no double-counting, "
                        "needs no special launch command or permissions "
                        "(see PssMemoryMonitor's docstring). 'cgroup': "
                        "Linux only, a single real number read from the "
                        "kernel's own cgroup v2 memory.peak accounting -- "
                        "exact in principle but depends on this cgroup's "
                        "delegation being set up exactly right; prefer "
                        "'pss' unless you specifically need this (see "
                        "CgroupMemoryMonitor's docstring for the sharp "
                        "edges hit in practice). Falls back to psutil "
                        "with a warning if a backend's preconditions "
                        "aren't met.")
    return p


def load_dataset_from_args(args):
    if args.data == "real":
        data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Real data dir not found: {data_dir}. Pass --data synthetic "
                f"or --data-dir to point at a directory with "
                f"sphere_irdis_{{cube,pa,psf}}_3.fits.")
        return load_real_dataset(data_dir)
    if args.data == "epoch":
        data_dir = Path(args.data_dir) if args.data_dir else (REPO_ROOT / "secondepoch")
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Epoch data dir not found: {data_dir}. Pass --data-dir to "
                f"point at a directory with center_im-2.fits, rotnth.fits, "
                f"finalPSF.fits (e.g. <repo>/firstepoch).")
        return load_epoch_dataset(data_dir)
    T, H, W = (int(x) for x in args.shape.split(","))
    return make_synthetic_dataset(T, H, W, fwhm=args.fwhm, seed=args.seed)


def make_test_positions(H: int, W: int, npix: int, seed: int = 0,
                        sampling: str = "dense") -> np.ndarray:
    """
    (row, col) test positions as float64, matching the phi0s convention.

    sampling:
      "dense"  (default) - a contiguous square block centered on the image,
                sized to ~npix positions. Representative of a real
                full-field sweep's *local density* per radius ring — this
                is what FullPACOv2's radius-ring cache-sharing optimization
                actually benefits from. Measured on sph3: a 1600-position
                dense block gave 2.43x speedup vs the original design,
                while an equally-sized *random* scatter only gave ~1.1x,
                because scattered positions rarely share a radius bin.
      "random" - a uniform random subset of the full grid. Useful for
                stress-testing correctness across varied radii/edges
                cheaply, but is NOT representative of the ring-cache
                benefit — don't use it alone to judge that optimization.
      "full"   - every pixel of the native H x W grid (ignores npix).
    """
    xx, yy = np.meshgrid(np.arange(0, H), np.arange(0, W))
    all_pos = np.column_stack((xx.flatten(), yy.flatten())).astype(float)

    if sampling == "full" or npix is None or npix >= len(all_pos):
        return all_pos

    if sampling == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_pos), size=npix, replace=False)
        return all_pos[idx]

    if sampling == "dense":
        half = max(1, int(round(np.sqrt(npix) / 2)))
        cy, cx = H // 2, W // 2
        r0, r1 = max(0, cy - half), min(H, cy + half)
        c0, c1 = max(0, cx - half), min(W, cx + half)
        rows, cols = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1))
        return np.column_stack((rows.flatten(), cols.flatten())).astype(float)

    raise ValueError(f"Unknown sampling mode: {sampling!r}")


def arrays_equal(a1, b1, a2, b2, rtol=1e-10, atol=1e-12):
    a_ok = np.allclose(a1, a2, equal_nan=True, rtol=rtol, atol=atol)
    b_ok = np.allclose(b1, b2, equal_nan=True, rtol=rtol, atol=atol)
    detail = ""
    if not (a_ok and b_ok):
        detail = (f" max|da|={np.nanmax(np.abs(a1 - a2)):.3e} "
                  f"max|db|={np.nanmax(np.abs(b1 - b2)):.3e}")
    return a_ok and b_ok, detail


class ResultTable:
    """Collects rows and prints/writes a clean summary at the end."""

    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add(self, **kwargs):
        self.rows.append({c: kwargs.get(c, "") for c in self.columns})

    def print_table(self):
        widths = {c: max(len(c), *(len(str(r[c])) for r in self.rows)) if self.rows
                 else len(c) for c in self.columns}
        header = "  ".join(c.ljust(widths[c]) for c in self.columns)
        print(header)
        print("-" * len(header))
        for r in self.rows:
            print("  ".join(str(r[c]).ljust(widths[c]) for c in self.columns))

    def write_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.columns)
            w.writeheader()
            w.writerows(self.rows)
        print(f"\nWrote {path}")


def system_info():
    import platform
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024**3
    except Exception:
        ram_gb = float("nan")
    gpu = "none"
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        gpu = f"{dev.name if hasattr(dev, 'name') else 'CUDA device'} " \
              f"({dev.mem_info[1] / 1024**3:.1f} GiB VRAM)"
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(ram_gb, 1),
        "gpu": gpu,
    }


def print_system_info():
    info = system_info()
    print("=" * 70)
    print("System info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 70)
