"""
Minimal fake `cupy` module, delegating every array operation to NumPy,
with just enough of CuPy's device/memory-pool/stream API surface stubbed
out to let src/gpu.py's GPU-only code paths (GPUStatisticsBatcher,
GPUBatchDriver, GPUFrameShifter) run on a machine with no CUDA device at
all -- this development machine included.

This does NOT measure real GPU speed or memory, and cannot reproduce a
real CUDA OutOfMemoryError (NumPy arrays don't live in a separate 8 GB
VRAM pool) -- it exists purely to catch API-shape bugs (wrong axis, wrong
dtype, a CuPy-only method that doesn't exist, mismatched broadcasting) in
code that, before this smoke test existed, had never executed even once.
See benchmarks/bench_gpu_fake_smoke.py for the test driven against it,
and benchmarks/bench_gpu_stats.py for the real-hardware correctness +
speed check to run on a server with an actual CUDA device.

Usage: call `install()` as the very first thing in a script's __main__,
before anything imports `cupy` or `src.gpu` (the `try: import cupy`
inside src/gpu.py only runs once per process -- it must see this fake
module already in sys.modules, not the real missing package).
"""
import sys
import types

import numpy as np


class FakeOutOfMemoryError(MemoryError):
    """Stand-in for cupy.cuda.memory.OutOfMemoryError."""


class _FakeMemoryPool:
    def free_all_blocks(self) -> None:
        pass

    def used_bytes(self) -> int:
        return 0


class _FakeDevice:
    def __init__(self, free_bytes: int = 8 * 1024**3, total_bytes: int = 16 * 1024**3):
        self._free_bytes = free_bytes
        self._total_bytes = total_bytes

    @property
    def mem_info(self):
        return (self._free_bytes, self._total_bytes)

    def synchronize(self) -> None:
        pass


class _FakeStream:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def synchronize(self) -> None:
        pass


def install(free_bytes: int = 8 * 1024**3, total_bytes: int = 16 * 1024**3):
    """
    Build and register a fake `cupy` module tree in sys.modules.

    `free_bytes` controls the (fake) VRAM GPUBatchDriver.estimate_batch_size
    will see on every call -- set it small to exercise the small-batch /
    many-chunk code path without needing a real small-VRAM device.

    Returns the fake `cupy` module (equivalently, `sys.modules["cupy"]`
    afterwards) in case the caller wants to tweak `.cuda.Device` further.
    """
    fake_cupy = types.ModuleType("cupy")
    # Every array op src/gpu.py and src/math_utils.py use (asarray, einsum,
    # linalg.inv/pinv, full, nan, isnan, any, sum, where, ones_like, clip,
    # eye, empty_like, diagonal, outer, indices, fft, ...) is a plain NumPy
    # call under this fake -- delegate everything not explicitly overridden
    # below straight to NumPy's own namespace.
    for name in dir(np):
        if not name.startswith("_"):
            setattr(fake_cupy, name, getattr(np, name))

    fake_cupy.asnumpy = lambda x: np.asarray(x)

    fake_cuda = types.ModuleType("cupy.cuda")
    fake_cuda.Device = lambda: _FakeDevice(free_bytes, total_bytes)
    fake_cuda.Stream = _FakeStream

    fake_memory = types.ModuleType("cupy.cuda.memory")
    fake_memory.OutOfMemoryError = FakeOutOfMemoryError
    fake_cuda.memory = fake_memory
    fake_cuda.MemoryPool = _FakeMemoryPool

    fake_cupy.cuda = fake_cuda
    fake_cupy.get_default_memory_pool = lambda: _FakeMemoryPool()

    sys.modules["cupy"] = fake_cupy
    sys.modules["cupy.cuda"] = fake_cuda
    sys.modules["cupy.cuda.memory"] = fake_memory
    return fake_cupy
