import numpy as np
import os

from astropy.io import fits
from scipy.fft import fft2, ifft2, fftshift
from astropy.convolution import convolve_fft


def build_kernel(psf_source, psf_ref, shape, eps=1e-3):
    """
    Build a convolution kernel that transforms psf_source into psf_ref
    using Fourier methods.

    Parameters
    ----------
    psf_source : 2D array
        PSF of the original image.
    psf_ref : 2D array
        PSF of the reference resolution.
    shape : tuple
        Shape (Ny, Nx) of the final kernel/image.
    eps : float
        Regularization parameter to avoid division by zero.

    Returns
    -------
    kernel : 2D array
        Convolution kernel.
    """
    Ny, Nx = shape

    psf_s = np.zeros((Ny, Nx))
    psf_t = np.zeros((Ny, Nx))

    psf_s[:psf_source.shape[0], :psf_source.shape[1]] = psf_source
    psf_t[:psf_ref.shape[0], :psf_ref.shape[1]] = psf_ref

    F_s = fft2(psf_s)
    F_t = fft2(psf_t)

    F_kernel = F_t * np.conj(F_s) / (np.abs(F_s)**2 + eps)

    kernel = np.real(ifft2(F_kernel))
    kernel = fftshift(kernel)
    kernel /= np.sum(kernel)

    return kernel

def save_kernel(kernel, output_path, header=None):
    """
    Save a convolution kernel to a FITS file.

    Parameters
    ----------
    kernel : 2D array
        Convolution kernel.
    output_path : str
        Output FITS path.
    header : fits.Header or None
        Optional header to attach to the kernel.
    """
    hdu = fits.PrimaryHDU(kernel, header=header)
    hdu.writeto(output_path, overwrite=True)
    print(f"Kernel saved at {output_path}")


def apply_kernel(image, kernel):
    """
    Convolve an image with a given kernel.

    Parameters
    ----------
    image : 2D array
    kernel : 2D array

    Returns
    -------
    image_conv : 2D array
    """
    return convolve_fft(image, kernel, allow_huge=True, normalize_kernel=False)
    
    
def psf_match_datacube(
    cube_path,
    kernel_dir,
    ref_filter="F444W",
    output_path=None,
    overwrite=True):
    """
    Apply PSF matching to a datacube using precomputed convolution kernels.

    Each slice is convolved to match the PSF of a reference filter.

    Parameters
    ----------
    cube_path : str
        Path to input datacube FITS.

    kernel_dir : str
        Directory containing kernel FITS files.

    ref_filter : str
        Reference filter (e.g., "F444W").

    output_path : str or None
        Output FITS file. If None, adds '_psfmatched'.

    overwrite : bool
        Overwrite output file.

    Returns
    -------
    output_path : str
        Path to saved PSF-matched datacube.
    """

    # Output path
    if output_path is None:
        output_path = cube_path.replace(".fits", "_psfmatched.fits")

    # Load cube
    with fits.open(cube_path) as hdul:
        cube = hdul[0].data
        header = hdul[0].header.copy()

    nfilters, ny, nx = cube.shape
    cube_conv = np.zeros_like(cube)

    # Loop over filters
    for i in range(nfilters):

        filt = header.get(f"FILTER{i+1}")
        if filt is None:
            print(f"No FILTER{i+1} keyword, skipping.")
            cube_conv[i] = cube[i]
            continue

        filt = filt.strip()
        filt_lower = filt.lower()

        print(f"Processing {i+1}/{nfilters}: {filt}")

        # Skip reference filter
        if filt.upper() == ref_filter.upper():
            cube_conv[i] = cube[i]
            continue

        # Kernel path
        kernel_path = os.path.join(
            kernel_dir,
            f"kernel_{filt_lower}_to_{ref_filter.lower()}.fits")

        if not os.path.exists(kernel_path):
            print(f"Kernel not found for {filt}, skipping.")
            cube_conv[i] = cube[i]
            continue

        kernel = fits.getdata(kernel_path)

        # Convolution
        cube_conv[i] = apply_kernel(cube[i], kernel)

    # Header update
    header.add_history(
        f"PSF matched to {ref_filter} using FFT convolution kernels")

    # Save
    fits.PrimaryHDU(cube_conv, header=header).writeto(
        output_path,
        overwrite=overwrite)

    print(f"PSF-matched datacube saved at {output_path}")

    return output_path
