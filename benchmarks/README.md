# Benchmarks and validation scripts

Scripts behind the timing, memory and detection results of the accompanying paper. Run them from the repository root, e.g. `python benchmarks/round2_experiments.py null --datasets sph3 --cpu 8`; each script's docstring lists its options.

## Data

No data is included in this repository.

- **EIDC data (`challenge/subchallenge_1/`)**: the ADI datasets of subchallenge 1 of the Exoplanet Imaging Data Challenge (Cantalloube et al. 2020, Proc. SPIE 11448, 114485A), with the challenge's own file names (`sphere_irdis_cube_3.fits`, `sphere_irdis_pa_3.fits`, `sphere_irdis_psf_3.fits`, `sphere_irdis_pxscale_3.fits`, …). They are on Zenodo, [doi:10.5281/zenodo.3361544](https://doi.org/10.5281/zenodo.3361544) (`subchallenge_1.zip`, ~2 GB); unzip the files into that folder.
- **HR 8799 (SPHERE/IRDIS, ESO program 095.C-0298(C))**: used by `e10_full_frame.py` and `round2_experiments.py ablation`, passed with `--cube`, `--angles` and `--psf`; available from the ESO Science Archive, not distributed.
- **Synthetic cubes**: generated on the fly by `_common.make_synthetic_dataset` (random correlated noise, field rotation 0–60°); no files needed.

## Scripts

`tests/` holds the unit tests: `python -m pytest`.

| Script | What it produces | Data |
|---|---|---|
| `round2_experiments.py null` | Null S/N statistics on rolled (companion-free) sequences, per dataset | EIDC sph1–3, NIRC2 1–3 |
| `round2_experiments.py ablation` | VIP's numerical choices switched on in this code: null width and calibrated S/N of real and injected companions | EIDC sph3, HR 8799 |
| `review_experiments.py` | Injected companions at a calibrated threshold, Fast vs Full PACO; the twice-finer candidate grid | EIDC sph3 |
| `e1_null_test.py` | Null S/N maps of this implementation and VIP on one rolled sequence | EIDC sph3 |
| `e9_memory_limit.py` | Completion under a fixed memory limit L, this implementation vs VIP | EIDC sph3 or `--synthetic SIZE T` |
| `e10_full_frame.py` | Full 1024×1024 frame: time, memory and planet S/N | HR 8799 |
| `bench_vs_vippaco.py` | Time and memory across worker counts, this implementation vs VIP, each in its own process | `--data`† |
| `bench_full_cube_comparison.py` | Shared-memory cache with and without ring traversal, and the earlier Manager-dict design (wall time and de-duplicated memory, ring vs native order; nproc sweep) | `--data`† |
| `bench_ring_landscape.py` | Run time over cube sizes and frame counts (validation of the time-complexity model) | synthetic |
| `bench_gpu_stats.py` | GPU statistics stage vs CPU (needs CuPy and a CUDA device) | `--data`† |

† `--data real` (default: EIDC sph3) or `--data synthetic`.

`challenge/` holds the scripts for the EIDC detection results: `rerun_sph3_challenge.py` (Full and Fast PACO S/N maps of sph3), `plot_challenge_results.py` (detection map, TPR/FDR curves and metrics), `run_vippaco_sph3.py` (VIP's FastPACO on the same data), and `sph3_data.py` (shared data loading).

Multiprocessing uses the `fork` start method, so the parallel benchmarks require Linux or macOS. Peak memory is measured with `psutil` by default, which counts pages shared between processes once per process; on Linux, `--memory-backend pss` gives a de-duplicated measurement.
