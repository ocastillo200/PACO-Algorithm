#!/usr/bin/env python3
"""
Local smoke test for the GPU statistics pipeline, using fake_cupy.py's
NumPy-backed fake `cupy` module -- run this BEFORE touching the server.

This machine has no CUDA device (GPU_AVAILABLE is False here), so the
entire GPUStatisticsBatcher / GPUBatchDriver code path, and all three of
its call sites (PACO.compute_statistics_gpu, FastPACOv2 Stage 1,
FullPACOv2's per-ring GPU cache prefill), had literally never executed
even once before this script existed -- not even enough to confirm the
array shapes line up. Faking `cupy` as a thin NumPy wrapper (installed
into sys.modules before anything imports it) lets that code run for
real, catching API-shape bugs for free, before spending server time
tracking down something catchable here.

What this does NOT do: measure real GPU speed, or trigger a real CUDA
OutOfMemoryError (NumPy arrays aren't allocated in a separate VRAM pool,
so "free_bytes" is just a number GPUBatchDriver divides by -- it can
force a tiny batch size, driving many small chunks through the same
code path, but it can't make a chunk actually fail to allocate). The
OOM-retry and per-chunk-CPU-fallback logic in GPUBatchDriver is instead
tested directly in Part 2 below, by handing it a `gpu_fn` that raises
`fake_cupy.FakeOutOfMemoryError` (or an arbitrary exception) on demand --
that logic has no CUDA dependency of its own, so this is a complete,
real test of it, not a fake one.

Usage
-----
    python3 benchmarks/bench_gpu_fake_smoke.py
    python3 benchmarks/bench_gpu_fake_smoke.py --npix 400 --shape 60,24,24
"""
import argparse
import sys

import fake_cupy
fake_cupy.install()  # MUST happen before any `import cupy` / `from src... import`

import numpy as np  # noqa: E402

from _common import arrays_equal, make_synthetic_dataset, make_test_positions  # noqa: E402

from src.gpu import GPU_AVAILABLE, GPUBatchDriver  # noqa: E402
from src.paco_base import PACO  # noqa: E402
from src.fast_paco import FastPACOv2  # noqa: E402
from src.full_paco import FullPACOv2  # noqa: E402


def part1_end_to_end(shape, npix, fwhm, seed):
    """Run all three GPU call sites against their CPU twin on a tiny
    synthetic cube, exactly as bench_gpu_stats.py does on real hardware --
    but here `use_gpu=True` routes through fake_cupy's NumPy backend."""
    print("=== Part 1: end-to-end correctness (fake GPU vs. CPU) ===")
    assert GPU_AVAILABLE, "fake_cupy.install() should have made GPU_AVAILABLE True"

    T, H, W = shape
    cube, pa, psf = make_synthetic_dataset(T, H, W, fwhm=fwhm, seed=seed)
    # PACO.compute_statistics_parallel (the base, non-Fast/Full variant
    # exercised below) always precomputes statistics for the FULL native
    # grid regardless of phi0s -- so phi0s must cover all H*W positions,
    # same convention as bench_gpu_stats.py's --npix default.
    npix = H * W if npix is None else npix
    phi0s = make_test_positions(H, W, npix, seed=seed, sampling="full")
    print(f"Dataset: cube={cube.shape}, fwhm={fwhm}, N positions={len(phi0s)}")

    all_ok = True

    p_cpu = PACO(cube, pa, psf, fwhm=fwhm, verbose=False)
    a_cpu, b_cpu = p_cpu.PACOCalc(phi0s, cpu=2, use_gpu=False)
    p_gpu = PACO(cube, pa, psf, fwhm=fwhm, verbose=False)
    a_gpu, b_gpu = p_gpu.PACOCalc(phi0s, cpu=2, use_gpu=True)
    ok, detail = arrays_equal(a_cpu, b_cpu, a_gpu, b_gpu)
    print(f"  [PACO/Algorithm1] correct={ok}{detail}")
    all_ok &= ok

    f_cpu = FastPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
    a_cpu2, b_cpu2 = f_cpu.PACOCalc(phi0s, cpu=2, use_gpu=False)
    f_gpu = FastPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
    a_gpu2, b_gpu2 = f_gpu.PACOCalc(phi0s, cpu=2, use_gpu=True)
    ok2, detail2 = arrays_equal(a_cpu2, b_cpu2, a_gpu2, b_gpu2)
    print(f"  [FastPACOv2]      correct={ok2}{detail2}")
    all_ok &= ok2

    fp_cpu = FullPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
    a_cpu3, b_cpu3 = fp_cpu.PACOCalc(phi0s, cpu=2, use_gpu=False)
    fp_gpu = FullPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
    a_gpu3, b_gpu3 = fp_gpu.PACOCalc(phi0s, cpu=2, use_gpu=True)
    ok3, detail3 = arrays_equal(a_cpu3, b_cpu3, a_gpu3, b_gpu3)
    print(f"  [FullPACOv2]      correct={ok3}{detail3}")
    all_ok &= ok3

    # Deliberately tiny fake VRAM: forces batch_size down to a handful of
    # items per chunk, exercising the multi-chunk loop in
    # compute_stats_streamed/compute_stage1_streamed (a single-chunk run
    # at generous VRAM would never touch that loop more than once).
    fake_cupy.install(free_bytes=1 * 1024**2)  # 1 MiB "VRAM"
    fp_gpu_tiny = FullPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
    a_tiny, b_tiny = fp_gpu_tiny.PACOCalc(phi0s, cpu=2, use_gpu=True)
    ok4, detail4 = arrays_equal(a_cpu3, b_cpu3, a_tiny, b_tiny)
    print(f"  [FullPACOv2, tiny fake VRAM, forces many small chunks] "
          f"correct={ok4}{detail4}")
    all_ok &= ok4
    fake_cupy.install()  # restore generous default for anything after this

    return all_ok


def part2_batch_driver_fault_injection():
    """Directly test GPUBatchDriver's OOM-retry and per-chunk CPU-fallback
    logic, with no CUDA/array dependency at all -- a gpu_fn that fails on
    demand is enough to exercise every branch."""
    print("\n=== Part 2: GPUBatchDriver fault injection (no array math) ===")
    import cupy as cp  # the fake module installed above
    all_ok = True

    # 2a. Every chunk OOMs until it shrinks to 1 item, then succeeds --
    # checks the halving loop terminates and still covers every item.
    driver = GPUBatchDriver(gpu_memory_safety_factor=0.8, max_retries=5, verbose=True)
    seen = []

    def gpu_fn_shrink_to_one(start, end):
        if (end - start) > 1:
            raise cp.cuda.memory.OutOfMemoryError()
        seen.append((start, end))

    def cpu_fn_should_not_run(start, end):
        raise AssertionError("cpu_fn should not be needed once chunk size reaches 1")

    driver.run(n_items=7, bytes_per_item=1.0, gpu_fn=gpu_fn_shrink_to_one,
              cpu_fn=cpu_fn_should_not_run)
    covered = sorted(seen)
    ok_a = covered == [(i, i + 1) for i in range(7)]
    print(f"  [2a] shrink-to-1-then-succeed: covered={covered} ok={ok_a}")
    all_ok &= ok_a

    # 2b. Every chunk OOMs no matter how small -- must exhaust retries and
    # fall back to cpu_fn for every item, never silently dropping one.
    driver2 = GPUBatchDriver(gpu_memory_safety_factor=0.8, max_retries=2, verbose=True)
    cpu_seen = []

    def gpu_fn_always_oom(start, end):
        raise cp.cuda.memory.OutOfMemoryError()

    def cpu_fn_record(start, end):
        cpu_seen.append((start, end))

    driver2.run(n_items=5, bytes_per_item=1.0, gpu_fn=gpu_fn_always_oom,
               cpu_fn=cpu_fn_record)
    total_covered = sum(e - s for s, e in cpu_seen)
    ok_b = total_covered == 5
    print(f"  [2b] always-OOM -> full CPU fallback: chunks={cpu_seen} "
          f"total_covered={total_covered} ok={ok_b}")
    all_ok &= ok_b

    # 2c. A non-OOM exception (e.g. a real CUDA driver error) falls back
    # immediately, without retrying.
    driver3 = GPUBatchDriver(gpu_memory_safety_factor=0.8, max_retries=3, verbose=True)
    gpu_calls = []
    cpu_seen3 = []

    def gpu_fn_weird_error(start, end):
        gpu_calls.append((start, end))
        raise RuntimeError("pretend CUDA driver error")

    def cpu_fn_record3(start, end):
        cpu_seen3.append((start, end))

    driver3.run(n_items=4, bytes_per_item=1.0, gpu_fn=gpu_fn_weird_error,
               cpu_fn=cpu_fn_record3)
    ok_c = len(gpu_calls) == 1 and sum(e - s for s, e in cpu_seen3) == 4
    print(f"  [2c] non-OOM error -> immediate CPU fallback, no retry: "
          f"gpu_calls={gpu_calls} cpu_chunks={cpu_seen3} ok={ok_c}")
    all_ok &= ok_c

    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", default="40,24,24",
                        help="T,H,W synthetic cube shape (default: 40,24,24 -- "
                             "deliberately tiny, this is a correctness smoke "
                             "test, not a speed benchmark).")
    parser.add_argument("--npix", type=int, default=None,
                        help="Default: full native grid (H*W) -- required "
                             "for the base PACO (non-Fast/Full) call site, "
                             "which always precomputes the whole grid.")
    parser.add_argument("--fwhm", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))

    ok1 = part1_end_to_end(shape, args.npix, args.fwhm, args.seed)
    ok2 = part2_batch_driver_fault_injection()

    print("\n=== Summary ===")
    print(f"Part 1 (end-to-end, fake GPU vs CPU): {'PASS' if ok1 else 'FAIL'}")
    print(f"Part 2 (GPUBatchDriver fault injection): {'PASS' if ok2 else 'FAIL'}")
    if not (ok1 and ok2):
        sys.exit(1)
    print("\nAll smoke tests passed. This does NOT confirm real GPU "
          "speed or memory behavior -- run bench_gpu_stats.py on the "
          "server for that.")


if __name__ == "__main__":
    main()
