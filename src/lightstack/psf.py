import numpy as np
import os

from astropy.io import fits
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from astropy.convolution import convolve_fft
from photutils.centroids import centroid_com
from scipy.ndimage import shift, zoom

from .utils import find_ext

def centroid_weighted(psf, threshold=1e-4):
    """
    Compute a flux-weighted centroid of a PSF, ignoring low-level noise.

    Parameters
    ----------
    psf : 2D ndarray
        Input PSF image.
    threshold : float, optional
        Fraction of the maximum PSF value below which pixels are ignored.
        Default is 1e-4.

    Returns
    -------
    ycen, xcen : float
        Coordinates of the centroid (in pixel units).
    """
    psf_copy = psf.copy()

    mask = psf_copy < threshold * psf_copy.max()
    psf_copy[mask] = 0

    return centroid_com(psf_copy)
    

def make_odd(psf):
    """
    Ensure that a PSF has odd dimensions by padding with zeros if necessary.

    Parameters
    ----------
    psf : 2D ndarray
        Input PSF image.

    Returns
    -------
    psf_padded : 2D ndarray
        PSF with odd dimensions.
    """
    ny, nx = psf.shape

    pad_y = 1 if ny % 2 == 0 else 0
    pad_x = 1 if nx % 2 == 0 else 0

    pad_before_y = pad_y // 2
    pad_after_y  = pad_y - pad_before_y

    pad_before_x = pad_x // 2
    pad_after_x  = pad_x - pad_before_x

    psf_padded = np.pad(
        psf,
        ((pad_before_y, pad_after_y),
         (pad_before_x, pad_after_x)),
        mode='constant',
        constant_values=0)

    return psf_padded


def resample_psf(
    input_path,
    output_path,
    zoom_factor=None,
    psf_pixel_scale=None,
    target_pixel_scale=None,
    order=3,
    normalize=True,
    make_odd_shape=True,
    header_comment=True):
    """
    Resample a PSF to match a target pixel scale, by directly providing a zoom factor
    or by specifying the original and target pixel scales. Always downsample to the worst resolution. 
    This code has not been tested for upsampling, so it is not recommended!

    Parameters
    ----------
    input_path : str
        Path to input PSF FITS file.
    output_path : str
        Path to save the resampled PSF.
    zoom_factor : float, optional
        Zoom factor to apply. If None, it will be computed from pixel scales.
    psf_pixel_scale : float, optional
        Original PSF pixel scale (arcsec/pixel).
    target_pixel_scale : float, optional
        Target pixel scale (arcsec/pixel).
    order : int, optional
        Interpolation order for scipy.ndimage.zoom. Default is 3 (cubic).
    normalize : bool, optional
        If True, normalize PSF to unit sum. Default is True.
    make_odd_shape : bool, optional
        If True, pad PSF to have odd dimensions. Default is True.
    header_comment : bool, optional
        If True, add history information to FITS header.

    Returns
    -------
    psf_resampled : 2D ndarray
        Resampled PSF array.
    """

    # Load PSF
    with fits.open(input_path) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No valid image extension found in {input_path}")

        psf = hdul[ext].data.astype(float)
        header = hdul[ext].header.copy()

    # Determine zoom factor
    if zoom_factor is None:
        if psf_pixel_scale is None or target_pixel_scale is None:
            raise ValueError("Provide either zoom_factor OR both psf_pixel_scale and target_pixel_scale")
        zoom_factor = psf_pixel_scale / target_pixel_scale

    if zoom_factor > 1:
    	raise ValueError("Upsampling not recommended.")
    
    # Resample
    psf_resampled = zoom(psf, zoom_factor, order=order)
    
    # Area correction
    psf_resampled /= zoom_factor**2

    # Ensure odd shape
    if make_odd_shape:
        psf_resampled = make_odd(psf_resampled)

    # Normalize
    if normalize:
        total = psf_resampled.sum()
        if total != 0:
            psf_resampled /= total

    # Update header
    if header_comment:
        header["HISTORY"] = "PSF resampled using scipy.ndimage.zoom"
        if target_pixel_scale is not None:
            header["CDELT1"] = (target_pixel_scale / 3600, "deg/pix")
            header["CDELT2"] = (target_pixel_scale / 3600, "deg/pix")

    # Save
    fits.PrimaryHDU(psf_resampled, header=header).writeto(
        output_path, overwrite=True)

    return psf_resampled



def build_kernel(psf_source, psf_ref, shape=(101, 101), eps=1e-3):
    """
    Build a convolution kernel that transforms psf_source into psf_ref
    using Fourier fast transforms.

    Parameters
    ----------
    psf_source : 2D array
        PSF of the original image.
    psf_ref : 2D array
        PSF of the reference resolution.
    shape : tuple
        Shape (Ny, Nx) of the kernel.
    eps : float
        Regularization parameter.

    Returns
    -------
    kernel : 2D array
        Convolution kernel.
    """

    Ny, Nx = shape

    psf_s = np.zeros((Ny, Nx))
    psf_t = np.zeros((Ny, Nx))

    ys, xs = psf_source.shape
    yt, xt = psf_ref.shape

    psf_s[
        Ny//2 - ys//2 : Ny//2 - ys//2 + ys,
        Nx//2 - xs//2 : Nx//2 - xs//2 + xs] = psf_source

    psf_t[
        Ny//2 - yt//2 : Ny//2 - yt//2 + yt,
        Nx//2 - xt//2 : Nx//2 - xt//2 + xt] = psf_ref

    psf_s = ifftshift(psf_s)
    psf_t = ifftshift(psf_t)

    # FFTs
    F_s = fft2(psf_s)
    F_t = fft2(psf_t)

    # Kernel construction (Wiener-like)
    F_kernel = F_t * np.conj(F_s) / (np.abs(F_s)**2 + eps)

    kernel = np.real(ifft2(F_kernel))
    kernel = fftshift(kernel)

    # Normalize
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
