# PACO-Algorithm

This repository contains a Python implementation of the PACO (PAtch COvariance) algorithm for exoplanet detection in high-contrast imaging data, as described in the paper by Flasseur et al. (2018).

This implementation includes the full, statistically robust version of the algorithm, an accelerated version for faster processing, and additional optimizations for modern hardware.

## Summary of the PACO Algorithm
To address the challenges of High Contrast Imaging (HCI), the PACO algorithm tackles the problem of **global non-stationarity**. This issue arises because the background, dominated by stellar speckles, has statistical properties that change across the image. PACO's solution is to divide the field of view into small, circular **patches** and assume local stationarity within them. This enables the algorithm to learn a precise and unique statistical model of the background—including its mean and spatial correlations (covariance)—for each individual location across time frames in ADI cubes.

By building a precise local model of the noise, PACO can perform a statistically robust hypothesis test to determine if a signal is present, leading to:

- **Statistically Grounded Detection Maps**: The resulting Signal-to-Noise Ratio (SNR) map has a consistent false alarm rate across all angular separations.
- **Unbiased Photometry**: The algorithm provides an unbiased estimate of the companion's flux without the self-subtraction issues common in other methods.
- **Parameter-Free Detection**: It requires minimal parameter tuning; the only parameter you should worry about is the patch size. By default the patch diameter is about twice the FWHM of the PSF (the initial guess proposed by the authors), but tuning it can noticeably improve detection performance.

## About This Implementation

This project provides a class-based implementation of the PACO algorithm, structured around the `vip_hci` library. It was originally derived from VIP's PACO module (`vip_hci.invprob.paco`) and has since been substantially rewritten; it keeps VIP's class interface and depends on `vip_hci` at run time. See `LICENSE` for VIP's copyright notice.

The key implemented components are:

- **`PACO`**: An implementation of **Algorithm 1** from the paper (called full PACO where it must be told apart from the fast variant). It computes the background statistics of each pixel on demand while sweeping the candidate positions, and stores them in a cache shared by all worker processes, so each pixel's statistics are computed once and reused by every candidate whose trajectory crosses it. Candidates are swept ring by ring (constant angular radius), and the cache is emptied between rings, which keeps its memory footprint to one ring's worth of statistics.
- **`FastPACO`**: An implementation of **Algorithm 2** from the paper (fast PACO). It precomputes per-pixel quantities on the native pixel grid and interpolates them at each candidate's rotated position, which is much faster than Algorithm 1. We recommend it for detection.
- **Unbiased Flux Estimation**: The `estimate_flux_at` method implements **Algorithm 3** for iterative, unbiased photometric measurement of detected sources.
- **Memory-aware batching**: every large allocation is sized from a single RAM budget, `memory_fraction` (default 0.8) of the memory available when a run starts (inside a container or a SLURM job, the headroom below its cgroup memory limit, if that is smaller). A larger field takes longer, but does not need more memory than the budget; when the cache is full, statistics are recomputed instead of stored, which costs time but never changes the result. The shared cache lives in `/dev/shm`, which containers size separately from their memory limit (Docker's default is 64 MB; raise it with `--shm-size`); the cache is shrunk to fit it.
- **Hardware Acceleration**:
  - **Multiprocessing**: Utilizes multiple CPU cores to parallelize calculations. The parallel path uses the `fork` start method, so it requires Linux or macOS; on other platforms, use `cpu=1`.
  - **GPU Support (optional)**: With `cupy` installed, the statistics stage can run on an NVIDIA GPU (`use_gpu=True`). This helps on small cubes but does not yet scale to large ones; the rest of the pipeline runs on the CPU.

The former class names `FullPACOv2` and `FastPACOv2` still work as deprecated aliases.

### Differences from VIP's PACO module

The patch statistics follow Flasseur et al. (2018) exactly: the sample covariance is the 1/T estimate and the shrinkage weight sums the squared diagonal of the covariance only (`tests/test_statistics.py` checks both against the paper's equations). VIP's module halves the covariance and sums over the whole matrix, which widens its S/N map by about √2, so S/N values from the two codes are not directly comparable. Statistics are taken at the nearest pixel of each rotated position (VIP truncates).

### Performance

On a full 1024×1024 SPHERE/IRDIS frame of HR 8799 (120 frames, 32 processes), `PACO` ran in 11 min with 2.6 GB of memory and `FastPACO` in 85 s, against 4.8 h and 135 GB for VIP's `FastPACO`; all three detect the four planets.

## Installation

Requires Python 3.10 or later; tested with NumPy 2.2–2.5 and `vip_hci` 1.6.6–2.1.1.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ocastillo200/PACO-Algorithm.git
    cd PACO-Algorithm
    ```

2.  **Create and activate a virtual environment (optional):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the package:**
    ```bash
    pip install .            # or: pip install -e . to edit the code in place
    ```
    This installs it as `paco` (`from paco import PACO`) together with its dependencies. Inside the clone, the same code can also be imported as `src`, which the benchmark scripts and tests use; `pip install -r requirements.txt` installs only the dependencies.

4.  **(Optional) Install CuPy for GPU acceleration:**
    `pip install ".[gpu]"`, or follow the CuPy installation guide for your CUDA version: <https://docs.cupy.dev/en/stable/install.html>

## Usage

### Basic Example

The following snippet shows how to load data and run the `PACO` algorithm to generate SNR and flux maps.

```python
import numpy as np
from paco import PACO   # or FastPACO, same interface
from vip_hci.fits import open_fits
from vip_hci.fm import normalize_psf
from vip_hci.config import VLT_SPHERE_IRDIS

# 1. Load your data
cube = open_fits('path/to/your/center_im.fits')[0]
angles = open_fits('path/to/your/rotnth.fits')
psf = open_fits('path/to/your/median_unsat.fits')[0]

# SPHERE pipeline products (rotnth.fits) store the angles with the opposite
# sign to the one PACO expects: negate them, or the companions smear out
# instead of adding up.
angles = -angles

# 2. Normalize the PSF and get instrument parameters
psfn, flux_normalized, fwhm = normalize_psf(psf, full_output=True)
pxscale = VLT_SPHERE_IRDIS['plsc']

# 3. Initialize the PACO algorithm
paco = PACO(
    cube=cube,
    angle_list=angles,
    psf=psfn,
    fwhm=fwhm,
    pixscale=pxscale,
    memory_fraction=0.8,  # share of available RAM this run may use
    verbose=True
)

# 4. Run the detection
# The run() method returns snr_map, flux_map when full_output=True
snr_map, flux_map = paco.run(cpu=8, full_output=True)

# 5. Save or plot the results
np.save('output_snr_map.npy', snr_map)
```

## Tests and benchmarks

```bash
pip install ".[test]"
python -m pytest
```

The tests check the patch statistics against the equations of Flasseur et al. (2018), that the S/N map does not depend on the number of worker processes or on the size of the shared cache, and the memory-limit detection. The scripts that produce the results of the accompanying paper are in `benchmarks/` and `challenge/` (see `benchmarks/README.md`).

## License and citation

MIT; see `LICENSE`, which also carries the license of VIP, from whose PACO module this code was derived. If you use this code, please cite Flasseur et al. (2018, A&A 618, A138) and the accompanying paper (in preparation).
