# PACO-Algorithm

This repository contains a Python implementation of the PACO (PAtch COvariance) algorithm for exoplanet detection in high-contrast imaging data, as described in the paper by Flasseur et al. (2018).

This implementation includes the full, statistically robust version of the algorithm, an accelerated version for faster processing, and additional optimizations for modern hardware.

## Summary of the PACO Algorithm
To address the challenges of High Contrast Imaging (HCI), the PACO algorithm tackles the problem of **global non-stationarity**. This issue arises because the background, dominated by stellar speckles, has statistical properties that change across the image. PACO's solution is to divide the field of view into small, circular **patches** and assume local stationarity within them. This enables the algorithm to learn a precise and unique statistical model of the background—including its mean and spatial correlations (covariance)—for each individual location across time frames in ADI cubes.

By building a precise local model of the noise, PACO can perform a statistically robust hypothesis test to determine if a signal is present, leading to:

- **Statistically Grounded Detection Maps**: The resulting Signal-to-Noise Ratio (SNR) map has a consistent false alarm rate across all angular separations.
- **Unbiased Photometry**: The algorithm provides an unbiased estimate of the companion's flux without the self-subtraction issues common in other methods.
- **Parameter-Free Detection**: It requires minimal parameter tuning, the only parameter you should worry about is the patch size. By default the radius of the patch is set as twice the FWHM of the system (as an good initial guess proposed by the authors), but finding its optimal value can drastically improve the detection performance.

## About This Implementation

This project, provides a class-based implementation of the PACO algorithm, structured around the `vip_hci` library.

The key implemented components are:

- **`FullPACO`**: A full implementation of **Algorithm 1** from the paper. This is the most statistically accurate version but is computationally intensive. It processes each potential source location by recalculating statistics along its specific trajectory.
- **`FastPACO`**: An implementation of **Algorithm 2** from the paper. It accelerates detection by pre-calculating the statistical maps for the entire image and then interpolating, significantly reducing runtime.
- **Unbiased Flux Estimation**: The `estimate_flux_at` method implements **Algorithm 3** for iterative, unbiased photometric measurement of detected sources.
- **Hardware Acceleration**:
  - **Multiprocessing**: Utilizes multiple CPU cores to parallelize calculations.
  - **GPU Support**: Includes a GPU-accelerated class using `cupy` for faster PSF model calculations, only works with `FullPACO` and It performs better in single core execution (no multiprocessing), further GPU support across the algorithm could be implemented to improve the performance even more.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/PACO-Algorithm.git](https://github.com/your-username/PACO-Algorithm.git)
    cd PACO-Algorithm
    ```

2.  **Create and activate a virtual environment (optional):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Install CuPy for GPU acceleration:**
    If you have an NVIDIA GPU and the CUDA toolkit installed, you can install CuPy for performance boost. Follow the official installation guide for your specific CUDA version: <https://docs.cupy.dev/en/stable/install.html>

## Usage

### Basic Example

The following snippet shows how to load data and run the `FullPACOv2` algorithm to generate SNR and flux maps.

```python
import numpy as np
from pacov2 import FullPACOv2
from vip_hci.fits import open_fits
from vip_hci.fm import normalize_psf
from vip_hci.config import VLT_SPHERE_IRDIS

# 1. Load your data
cube = open_fits('path/to/your/center_im.fits')[0]
angles = open_fits('path/to/your/rotnth.fits')
psf = open_fits('path/to/your/median_unsat.fits')[0]

# 2. Normalize the PSF and get instrument parameters
psfn, flux_normalized, fwhm = normalize_psf(psf, full_output=True)
pxscale = VLT_SPHERE_IRDIS['plsc']

# 3. Initialize the PACO algorithm
paco = FullPACOv2(
    cube=cube,
    angle_list=angles,
    psf=psfn,
    fwhm=fwhm,
    pixscale=pxscale,
    verbose=True
)

# 4. Run the detection
# The run() method returns snr_map, flux_map when full_output=True
snr_map, flux_map = paco.run(cpu=8, full_output=True)

# 5. Save or plot the results
np.save('output_snr_map.npy', snr_map)
