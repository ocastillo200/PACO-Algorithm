# Benchmarks and validation scripts

Scripts behind the timing, memory and detection results of the accompanying paper. Run them from the repository root, e.g. `python benchmarks/e1_null_test.py --cpu 8`; each script's docstring lists its options.

## Data

No data is included in this repository.

- **EIDC data (`challenge/subchallenge_1/`)**: the ADI datasets of subchallenge 1 of the Exoplanet Imaging Data Challenge (Cantalloube et al. 2020, Proc. SPIE 11448, 114485A), with the challenge's own file names (`sphere_irdis_cube_3.fits`, `sphere_irdis_pa_3.fits`, `sphere_irdis_psf_3.fits`, `sphere_irdis_pxscale_3.fits`, …). They are on Zenodo, [doi:10.5281/zenodo.3361544](https://doi.org/10.5281/zenodo.3361544) (`subchallenge_1.zip`, ~2 GB); unzip the files into that folder.
- **Real-instrument epochs (`firstepoch/`, `secondepoch/`)**: used only by `challenge/run_real_system_comparison.py` and the `load_epoch_dataset` helper; not distributed.
- **Synthetic cubes**: generated on the fly by `_common.make_synthetic_dataset` (random correlated noise, field rotation 0–60°); no files needed.

## Scripts

`tests/test_cgroup_headroom.py` tests the cgroup-aware memory budget (`python -m pytest tests/`).

| Script | What it measures | Data |
|---|---|---|
| `e1_null_test.py` | Null-distribution test: S/N statistics on a companion-free (sign-reversed angles) sequence, this implementation vs VIP; `--check-covariance` compares the covariance estimates directly | EIDC sph3 |
| `e9_memory_limit.py` | Completion under a fixed memory limit L: each run is killed if its process tree exceeds L (PSS on Linux); this implementation vs VIP, with a bit-for-bit check of our completed runs | EIDC sph3 or `--synthetic SIZE T` |
| `bench_ring_speedup.py` | Speed-up and efficiency vs number of worker processes, fixed cube | synthetic |
| `bench_ring_size_scaling.py` | Run time and peak memory vs cube size, fixed number of workers | synthetic |
| `bench_ring_landscape.py` | Run time / memory over a grid of cube sizes and worker counts | synthetic |
| `bench_full_cube_comparison.py` | Earlier Full PACO designs vs the shared-memory cache, with and without ring traversal | `--data`† |
| `bench_vs_vippaco.py` | This implementation vs VIP's PACO: time and peak memory across worker counts | `--data`† |
| `bench_full_paco.py`, `bench_fast_paco.py`, `bench_paco_base.py` | Earlier vs current design of each class, with an output-equality check | `--data`† |
| `bench_gpu_stats.py` | GPU statistics stage vs CPU (needs CuPy and a CUDA device) | `--data`† |
| `bench_gpu_fake_smoke.py` + `fake_cupy.py` | Exercises the GPU code paths on a machine without a GPU, using a NumPy-backed stand-in for CuPy | synthetic |

† `--data real` (default: EIDC sph3), `--data synthetic`, or `--data epoch` (real-instrument epochs).

`challenge/` holds the detection-performance scripts: `run_paco_challenge.py` / `rerun_sph3_challenge.py` (EIDC detection maps), `plot_challenge_results.py` (figures and metrics), `make_contrast_curve_sph3.py` / `run_contrast_curve_server.py` / `calibrate_contrast_units.py` (contrast curves), `run_vippaco_sph3.py` (VIP's FastPACO on the same data) and `run_real_system_comparison.py`. `challenge/scripts/` contains helper scripts for the challenge's detection-map format.

Multiprocessing uses the `fork` start method, so the parallel benchmarks require Linux or macOS. Peak memory is measured with `psutil` by default, which counts pages shared between processes once per process; on Linux, `--memory-backend pss` gives a de-duplicated measurement.
