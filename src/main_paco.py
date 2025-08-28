import sys
import numpy as np
from abc import abstractmethod
from typing import List, Optional, Union, Tuple, Callable
from vip_hci.preproc.rescaling import frame_shift
from vip_hci.preproc.rescaling import cube_px_resampling
from vip_hci.preproc.rescaling import frame_px_resampling
from math_functions import compute_statistics_at_pixel
from utils import create_boolean_circular_mask, get_rotated_pixel_coords


class PACO:
    """
    This class implements the bulk of the PACO algorithm as described in
    [FLA18]_. In general, the idea is to take in an ADI stack of images and
    statistically determine if there is a signal above the background in each
    'patch' of the image. This is done by tracing the ark of the hypothesized
    planet through the stack, and comparing this set of patches to a set
    consisting of background only. This is done for each pixel (or sub-pixel)
    location in the image. The output is a signal-to-noise and/or a flux map
    over the field of view. The user can choose to use FullPACO or FastPACO,
    which are described by algorithms 1 and 2 of [FLA18]_. FastPACO has been
    parallelized, and is the recommended usage.

    This output can then be used to compute an unbiased estimate of the flux
    of point sources detected in the image above some user-supplied detection
    threshold.

    Parameters
    ----------
    cube : numpy.ndarray
        3D science frames taken in pupil tracking/ADI mode.
        Dimensions should be (time, x, y), and units should be detector units
        (ie output of SPHERE or GPI reduction pipelines). The data should be
        centered, and have pre-processing already applied (e.g. bad pixel
        correction).
    angles : numpy.ndarray
        List of derotation angles for each frame in degrees. Length of this
        array should be the same as the time axis of the science cube.
    psf : numpy.ndarray
        Unsaturated PSF image. If a cube is provided, the median of the cube
        will be used.
    dit_psf : float, optional
        Integration time of the unsaturated PSF in seconds. The PSF is
        normalised to dit_science/dit_psf/nd_transmission.
    dit_science : float, optional
        Integration time of the science frames in seconds. The PSF is normalised
        to dit_science/dit_psf/nd_transmission.
    nd_transmission : float, optional
        Transmission of an ND filter used to aquire the unsaturated PSF. The PSF
        is normalised to dit_science/dit_psf/nd_transmission.
    fwhm : float, optional
        FWHM of PSF in arcseconds. Default values give 4px radius.
    pixscale : float, optional
        Detector pixel scale in arcseconds per pixel.  Default values give 4px
        radius.
    rescaling_factor : float, optional
        Scaling for sub/super pixel resolution for PACO. Will rescale both the
        science cube and the PSF.
    verbose : bool, optional
        Sets level of printed outputs.
    """

    def __init__(self,
                 cube: np.ndarray,
                 angle_list: np.ndarray,
                 psf: np.ndarray,
                 dit_psf: Optional[float] = 1.0,
                 dit_science: Optional[float] = 1.0,
                 nd_transmission: Optional[float] = 1.0,
                 fwhm: Optional[float] = 4.0,
                 pixscale: Optional[float] = 1.0,
                 rescaling_factor: Optional[float] = 1.0,
                 use_gpu: Optional[bool] = False,
                 flux_normalized: Optional[float] = 1.0,
                 full_output: Optional[bool] = False,
                 verbose: Optional[bool] = False) -> None:

        # Science image setup
        try:
            self.cube = cube
        except BaseException:
            raise ValueError("You must provide a 3D cube of science data!")

        self.num_frames = self.cube.shape[0]
        self.width = self.cube.shape[2]
        self.height = self.cube.shape[1]

        # Parallactic angles
        try:
            self.angles = angle_list
        except BaseException:
            raise ValueError("You must provide an array of parallactic angles!")

        # Pixel scaling
        self.pixscale = pixscale
        self.rescaling_factor = rescaling_factor
        self.full_output = full_output

        # PSF setup
        if isinstance(fwhm, (list, np.ndarray)):
            self.fwhm = int(np.median(fwhm))
        else:
            self.fwhm = int(fwhm)

        try:
            # Taking the median for stack of psfs, could be a single psf aribitrarly chosen based on the quality observed
            if len(psf.shape) > 2:
                psf = np.nanmedian(psf, axis=0)
            self.psf = psf * dit_science/dit_psf/nd_transmission
            self.dit_science = dit_science
            self.dit_psf = dit_psf
            self.flux_normalized = flux_normalized

            mask = create_boolean_circular_mask(self.cube[0].shape,
                                                radius=self.fwhm)
            self.patch_area_pixels = self.cube[0][mask].ravel().shape[0]
        except BaseException:
            raise ValueError("You must provide an unsaturated PSF image!")

        self.patch_width = 2*int(self.fwhm) + 1
        self.verbose = verbose
        self.use_gpu = use_gpu

        # These are what we're calculating
        self.snr = None
        self.flux = None
        self.std = None

        # Diagnostics
        if self.verbose:
            print("---------------------- ")
            print("Summary of PACO setup: \n")
            print(f"Image Cube shape = {self.cube.shape}")
            print("PSF |  Area  |  Rad   |  Width | ")
            print(f"    |   {self.patch_area_pixels:01}   |"
                  + f"  {self.fwhm:02}    |  {self.psf.shape[0]:03}   | ")
            print(f"Patch width: {self.patch_width}")
            print("---------------------- \n")
            sys.stdout.flush()

    @abstractmethod
    def PACOCalc(self,
                 phi0s: np.ndarray,
                 use_subpixel_psf_astrometry: Optional[bool] = True,
                 cpu: Optional[int] = 1) -> None:
        """
        This function is algorithm dependant, and sets up the actual
        calculation process.

        Parameters
        ----------
        phi0s : numpy.ndarray
            Array of pixel coordinates to try to search for the planet signal.
            Typically a grid created using numpy.meshgrid.
        use_subpixel_psf_astrometry : bool
            If true, the PSF model for each patch is shifted to the correct
            location as predicted by the starting location and the parallactic
            angles, before being resampled for the patch. If false, the PSF
            model is simply located at the center of each patch. Significantly
            improves performance if set to False, but the SNR is reduced.
        cpu : int, optional
            Number of cpus to use for parallelization.

        Returns
        -------
        a : numpy.ndarray
            a_l from Equation 15 of [FLA18]_
        b : numpy.ndarray
            b_l from Equation 16 of [FLA18]_
        """

    def run(self,
            cube: Optional[np.ndarray] = None, 
            angle_list: Optional[np.ndarray] = None, 
            cpu: Optional[int] = 1,
            imlib: Optional[str] = 'vip-fft',
            interpolation: Optional[str] = 'lanczos4',
            keep_center: Optional[bool] = True,
            verbose: Optional[bool] = None,
            use_gpu: Optional[bool] = False,
            full_output: Optional[bool] = False,
            use_subpixel_psf_astrometry: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run method of the PACO class. This function wraps up the PACO
        calculation steps, and returns the snr and flux estimates as
        outputs. The image library arguments are used if a rescaling of the
        data is desired: upsampling the images will result in a better SNR
        detection, but will be slower.
        
        Parameters
        ----------
        cpu : int, optional
            Number of processers to use
        imlib : str, optional
            See the documentation of the ``vip_hci.preproc.frame_px_resampling``
            function.
        interpolation : str, optional
            See the documentation of the ``vip_hci.preproc.frame_px_resampling``
            function.
        keep_center: bool, opt
            If input dimensions are even and the star centered (i.e. on
            dim//2, dim//2), whether to keep the star centered after scaling, i.e.
            on (new_dim//2, new_dim//2). For a non-centered input cube, better to
            leave it to False.
        use_subpixel_psf_astrometry : bool
            If true, the PSF model for each patch is shifted to the correct
            location as predicted by the starting location and the parallactic
            angles, before being resampled for the patch. If false, the PSF
            model is simply located at the center of each patch. Significantly
            improves performance if set to False, but the SNR is reduced.

        Returns
        -------
        snr : numpy.ndarray
            2D map of the signal-to-noise estimate as computed by PACO.
            This is b/sqrt(a), as in eqn 24 of [FLA18]_.
        flux : numpy.ndarray
            2D map of the flux estimate as computed by PACO
            This is b/a, as in eqn 21 of [FLA18]_.
        """
        if verbose is not None:
            self.verbose = verbose

        if cube is not None and angle_list is not None:
            self.set_cube(cube)
            self.set_angles(angle_list)

        if full_output is not None:
            self.full_output = full_output

        if self.rescaling_factor != 1:
            self.rescale_cube_and_psf(imlib=imlib,
                                      interpolation=interpolation,
                                      keep_center=keep_center,
                                      verbose=self.verbose)
            if self.verbose:
                print("---------------------- ")
                print(f"Using {cpu} processor(s).")
                print(f"Rescaled Image Cube shape: {self.cube.shape}")
                print("Rescaled PSF:")
                print("PSF |  Area  |  Rad   |  Width | ")
                print(f"    |   {self.patch_area_pixels:01}   |"
                      + f"  {self.fwhm:02}    |  {self.psf.shape[0]:03}   | ")
                print("---------------------- \n")

        # Setup pixel coordinates
        x, y = np.meshgrid(np.arange(0, self.height),
                           np.arange(0, self.width))
        phi0s = np.column_stack((x.flatten(), y.flatten()))
        # Compute a,b
        a, b = self.PACOCalc(np.array(phi0s), cpu=cpu, use_gpu=use_gpu)

        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a, (self.height, self.width))
        b = np.reshape(b, (self.height, self.width))
        # Output arrays
        snr = b/np.sqrt(a)
        flux = b/a
        self.snr = snr
        self.flux = flux
        self.std = 1/np.sqrt(a)

        if self.full_output:
            return snr, flux
        else:
            return snr 

    """
    Utility Functions
    """
    # Set the image stack to be processed

    def set_cube(self, cube: np.ndarray) -> None:
        """
        Provide a 3D image array to process. This updates
        the science cube, and the associated dimensions.

        Parameters
        ----------
        cube : numpy.ndarray
            3D science frames taken in pupil tracking/ADI mode.
            Dimensions should be (time, x, y), and units should be detector
            units (ie output of SPHERE or GPI reduction pipelines). The data
            should be centered, and have pre-processing already applied (e.g.
            bad pixel correction).
        """
        self.cube = np.array(cube)
        self.num_frames = self.cube.shape[0]
        self.width = self.cube.shape[2]
        self.height = self.cube.shape[1]

    # Set the template PSF
    def set_psf(self, psf: np.ndarray) -> None:
        """
        Read in the PSF template

        Parameters
        ----------
        psf: numpy.ndarray
            An unsaturated psf to use as the template.
        """
        self.psf = psf

    # Set parallactic angles
    def set_angles(self, angles: np.ndarray) -> None:
        """
        Set the rotation angle for each frame

        Parameters
        ----------
        angles: numpy.ndarray
            A list of the parallactic angles for each frame of the science data.
        """

        self.angles = angles

    def get_patch(self,
              px: Tuple[int, int],
              width: Optional[int] = None,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Gets patch at given pixel px with size width and optional mask.

        Returns a (num_frames, patch_area_pixels) array.
        """
        if width is None:
            width = self.patch_width
        if mask is None:
            mask = create_boolean_circular_mask(self.cube[0].shape,
                                                radius=self.fwhm,
                                                center=px)

        k = width // 2
        k2 = k + (width % 2)
        x, y = px
        nx, ny = self.cube.shape[1:3]

        if not (k <= x < nx - k2 and k <= y < ny - k2):
            return np.full((self.num_frames, self.patch_area_pixels), np.nan)

        patch = self.cube[np.broadcast_to(mask, self.cube.shape)]
        patch = patch.reshape(self.num_frames, self.patch_area_pixels)
        return patch

    def set_scale(self, scale: float) -> None:
        """
        Set subpixel scaling factor

        Parameters
        ----------
        scale : float
            Scaling factor. Greater than one will result in an upsampled image,
            less than one will result in a downsampled image.
        """

        self.rescaling_factor = scale

    def rescale_cube_and_psf(self,
                             imlib: Optional[str] = 'vip-fft',
                             interpolation: Optional[str] = 'lanczos4',
                             keep_center: Optional[bool] = True) -> None:
        """
        Rescale each image in the stack by the class level scaling factor
        set during initialization or with set_scale. A scale factor of greater
        than one will upsample the image, a factor of less than one will downsample
        the image. This function wraps the VIP scaling function, and uses the same
        arguments to choose libraries and interpolation methods.

        Parameters
        ----------
        imlib : str, optional
            See the documentation of the ``vip_hci.preproc.frame_px_resampling``
            function.
        interpolation : str, optional
            See the documentation of the ``vip_hci.preproc.frame_px_resampling``
            function.
        keep_center: bool, opt
            If input dimensions are even and the star centered (i.e. on
            dim//2, dim//2), whether to keep the star centered after scaling, i.e.
            on (new_dim//2, new_dim//2). For a non-centered input cube, better to
            leave it to False.
        """

        if self.rescaling_factor == 1:
            if self.verbose:
                print("Scale is 1, no scaling applied.")
            return

        # Resample the science cube
        cube_px_resampling(self.cube,
                           self.rescaling_factor,
                           imlib=imlib,
                           interpolation=interpolation,
                           keep_center=keep_center,
                           verbose=False)

        self.pixscale = self.pixscale/self.rescaling_factor
        self.fwhm = int(self.fwhm*self.rescaling_factor)

        # Resample the PSF
        if self.psf is not None:
            self.psf = frame_px_resampling(self.psf,
                                           self.rescaling_factor,
                                           imlib=imlib,
                                           interpolation=interpolation,
                                           keep_center=keep_center,
                                           verbose=False)
        mask = create_boolean_circular_mask(self.psf.shape, self.fwhm)
        self.patch_area_pixels = self.psf[mask].shape[0]
        self.patch_width = 2*int(self.fwhm) + 3

    """
    Math Functions
    """

    def psf_model_function(self, mean: float, model: Callable,
                           params: dict) -> np.ndarray:
        """
        This function is deprecated in favour of directly supplying
        a PSF. In principle, an analytic model (ie a gaussian or moffat PSF)
        can be used in place of a measured unsaturated PSF.

        Parameters
        ----------
        mean : float
            If using the psfTemplateModel function, the mean
        model : dnc
            numpy statistical model (need to import numpy module for this)
        **kwargs: dict
            additional arguments for model

        Returns
        -------
        self.psf : numpy.ndarray
            Returns the PSF template used by PACO
        """

        if self.psf:
            return self.psf
        if model is None:
            print("Please input either a 2D PSF or a model function.")
            sys.exit(1)
        else:
            if model.__name__ == "psfTemplateModel":
                try:
                    self.psf = model(mean, params)
                    return self.psf
                except ValueError:
                    print("Fix template size")
            self.psf = model(mean, params)
            return self.psf

    def wfl(self,
        hfl: Union[List, np.ndarray],
        Cfl_inv: Union[List, np.ndarray]) -> np.ndarray:
        """
        Computes w_l = C_l^{-1} h_l for each time frame.

        Parameters
        ----------
        hfl : list or numpy.ndarray
            List of flattened PSF templates.
        Cfl_inv : list or numpy.ndarray
            List of inverse covariance matrices.

        Returns
        -------
        wfl : numpy.ndarray
            List of w_l vectors for each frame.
        """
        return np.array([Cfl_inv[i] @ hfl[i] for i in range(len(hfl))])


    def al(self,
        hfl: Union[List, np.ndarray],
        wfl: Union[List, np.ndarray]) -> float:
        """
        Computes the sum of a_l terms using the precomputed w_l = C^{-1}h.

        Parameters
        ----------
        hfl : list or numpy.ndarray
            List of flattened PSF templates.
        wfl : list or numpy.ndarray
            List of precomputed w_l vectors.

        Returns
        -------
        a : float
            The sum of a_l = h_l^T * w_l across all frames.
        """
        return np.sum([hfl[i] @ wfl[i].T for i in range(len(hfl))])


    def bl(self,
        wfl: Union[List, np.ndarray],
        r_fl: Union[List, np.ndarray],
        m_fl: Union[List, np.ndarray]) -> float:
        """
        Computes the sum of b_l terms using the precomputed w_l = C^{-1}h.

        Parameters
        ----------
        wfl : list or numpy.ndarray
            List of precomputed w_l vectors.
        r_fl : list or numpy.ndarray
            List of flattened data patches.
        m_fl : list or numpy.ndarray
            List of flattened mean background patches.

        Returns
        -------
        b : float
            The sum of b_l = w_l^T * (r_l - m_l) across all frames.
        """
        return np.sum([wfl[i].T @ (r_fl[i] - m_fl[i]) for i in range(len(wfl))])

    """
    FluxPACO
    """

    def estimate_flux_at(self,
                        phi0: Tuple[int, int],
                        initial_est: float = 0.0,
                        eps: float = 0.01,
                        max_iter: int = 50) -> Tuple[float, float, float]:
        """
        Estimates the unbiased flux of a SINGLE source at a given location,
        strictly following Algorithm 3 from [FLA18].
        """
        if self.verbose:
            print(f"\nEstimating unbiased flux for source at {phi0}...")
        
        norm = self.flux_normalized
        normalised_psf = self.psf
        psf_mask = create_boolean_circular_mask(normalised_psf.shape, radius=self.fwhm)

        dimx = self.width / 2
        grid_x, grid_y = np.meshgrid(np.arange(-dimx, dimx), np.arange(-dimx, dimx))

        p0_yx = (phi0[1], phi0[0])
        angles_px = get_rotated_pixel_coords(grid_x, grid_y, p0_yx, self.angles)
        
        h_on = []
        r_track_original = []
        for l, ang in enumerate(angles_px):
            ang_int = ang.astype(int)
            ang_subpix = ang - ang_int
            psf_shifted = frame_shift(normalised_psf, shift_x=ang_subpix[1], shift_y=ang_subpix[0])[psf_mask]
            h_on.append(psf_shifted)
            
            patch_col = self.get_patch((ang_int[1], ang_int[0]))
            r_track_original.append(patch_col[l] if patch_col is not None else np.full(self.patch_area_pixels, np.nan))
        
        h_on = np.array(h_on)
        r_track_original = np.array(r_track_original)

        ahat = initial_est
        aprev = np.inf
        iter_count = 0
        max_iters = max_iter

        convergence_history = []
        while np.abs(ahat - aprev) > ahat * eps and iter_count < max_iters:
            aprev = ahat
            iter_count += 1
            
            a_total, b_total = 0.0, 0.0
            stats_cache = {}

            for l, ang in enumerate(angles_px):
                theta_l = tuple(ang.astype(int))

                if theta_l in stats_cache:
                    m_u, Cinv_u = stats_cache[theta_l]
                else:
                    observed_patches_at_theta = self.get_patch((theta_l[1], theta_l[0]))
                    
                    if observed_patches_at_theta is None or np.all(np.isnan(observed_patches_at_theta)):
                        stats_cache[theta_l] = (None, None)
                        continue
                        
                    contaminant_signal = np.zeros_like(observed_patches_at_theta)
                    for l_prime, ang_prime in enumerate(angles_px):
                        shift_y = ang_prime[0] - theta_l[0]
                        shift_x = ang_prime[1] - theta_l[1]
                        contaminant_psf = frame_shift(normalised_psf, shift_x, shift_y)[psf_mask]
                        contaminant_signal[l_prime, :] = contaminant_psf
                    
                    unbiased_patches = observed_patches_at_theta - (ahat * contaminant_signal)
                    m_u, Cinv_u = compute_statistics_at_pixel(unbiased_patches)
                    stats_cache[theta_l] = (m_u, Cinv_u)

                current_r = r_track_original[l]
                current_h = h_on[l]

                if m_u is None or Cinv_u is None or np.any(np.isnan(current_r)):
                    continue
                
                w_l = Cinv_u @ current_h
                a_l = current_h.T @ w_l
                b_l = w_l.T @ (current_r - m_u)
                
                a_total += a_l
                b_total += b_l

            if a_total > 1e-8:
                ahat = max(0.0, b_total) / a_total
            else:
                ahat = 0.0
            
            convergence_history.append(ahat / norm)

            if self.verbose:
                print(f"  Iter {iter_count}: a={a_total:.2f}, b={b_total:.2f}, new flux_est={ahat/norm:.2f}")

        final_std = 1 / np.sqrt(a_total) if a_total > 1e-8 else np.inf
        final_flux = ahat / norm
        final_std_norm = final_std / norm
        
        if self.verbose:
            print(f"Final estimate for {phi0}: flux = {final_flux:.2f} ± {final_std_norm:.2f}")

        return final_flux, final_std_norm, norm, convergence_history
