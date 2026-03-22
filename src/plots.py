import numpy as np
import os
import glob

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm

from .utils import find_ext, infer_filter, get_filter


def visualize_fits(fits_path, save_path=None, stretch='log',
                   min_percent=25., max_percent=99.98,
                   cmap='viridis', xlim=None, ylim=None):
    """
    Visualizes a FITS file with both pixel and RA/Dec axes.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.
    save_path : str or None
        Path to save the output image. If None, the figure is shown but not saved.
    stretch : str
        Stretch type for simple_norm (e.g., 'linear', 'log', 'sqrt').
    min_percent, max_percent : float
        Percentile limits for normalization.
    cmap : str
        Colormap name.
    xlim, ylim : tuple or None
        Pixel axis limits.
    """
    # Open FITS file
    with fits.open(fits_path, memmap=False) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            print(f"No data extension found in {fits_path}")
            return

        data = hdul[ext].data
        header = hdul[ext].header
        wcs = WCS(header)

    # Mask invalid or non-positive values for normalization
    mask = np.logical_or(np.isnan(data), data <= 0.)
    norm = simple_norm(
        data[~mask],
        stretch=stretch,
        min_percent=min_percent,
        max_percent=max_percent)

    # Create figure
    plt.rcParams['font.size'] = 10
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': wcs})

    im = ax.imshow(
        data,
        cmap=cmap,
        origin='lower',
        norm=norm,
        interpolation='nearest')

    # World coordinates
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    # Pixel coordinates
    secax = ax.secondary_xaxis('top')
    secax.set_xlabel('x (pixel)')
    secay = ax.secondary_yaxis('right')
    secay.set_ylabel('y (pixel)')

    # Set limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.grid()

    # Use filename to build a simple title
    base = os.path.basename(fits_path)
    region = base.split('_')[0]
    ax.set_title(region)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved image at {save_path}")
        plt.close(fig)
    else:
        plt.show()
        
        
def plot_datacube_filters(
    cube_fits_file,
    ncols=None,
    figsize=(15, 15),
    cmap="viridis",
    norm=None,
    stretch="log",
    min_percent=25.,
    max_percent=99.98,
    save_path=None):
    """
    Plot all filters from a datacube as a grid of images.

    Parameters
    ----------
    cube_fits_file : str
        Path to the 3D FITS datacube.

    ncols : int or None
        Number of columns in the grid. If None, a near-square layout is used.

    figsize : tuple
        Figure size.

    cmap : str
        Colormap.

    norm : matplotlib.colors.Normalize or None
        Custom normalization. If None, uses simple_norm.

    stretch : str
        Stretch for simple_norm (ignored if norm is provided).

    min_percent, max_percent : float
        Percentile limits for normalization.

    save_path : str or None
        If provided, saves the figure.

    show : bool
        If True, displays the figure.
    """

    # Open FITS 
    with fits.open(cube_fits_file) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No valid data extension in {cube_fits_file}")

        datacube = hdul[ext].data
        header = hdul[ext].header

    # Filters
    n_filters = datacube.shape[0]

    filters = [header.get(f"FILTER{i+1}", f"{i}") for i in range(n_filters)]

    # Layout for figure
    if ncols is None:
        # Try to make a square-like grid
        ncols = int(np.ceil(np.sqrt(n_filters)))

    nrows = int(np.ceil(n_filters / ncols))

    # Figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_filters):
        ax = axes[i]
        data = datacube[i]

        mask = np.logical_or(np.isnan(data), data <= 0.)

        if norm is None:
            if np.all(mask):
                norm_i = None
            else:
                norm_i = simple_norm(
                    data[~mask],
                    stretch=stretch,
                    min_percent=min_percent,
                    max_percent=max_percent)
        else:
            norm_i = norm

        ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            norm=norm_i,
            interpolation="nearest")

        ax.set_title(f"{filters[i]}")
        ax.axis("off")

    for j in range(n_filters, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved image at {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_psf_grid(
    psf_dir=None,
    psf_files=None,
    ncols=None,
    figsize=(12, 12),
    norm=None,
    stretch="log",
    percent=99.0,
    cmap="viridis",
    save_path=None):
    """
    Plot a grid of PSFs from FITS files.

    Parameters
    ----------
    psf_dir : str, optional
        Directory containing PSF FITS files

    psf_files : list, optional
        List of PSF FITS file paths (overrides psf_dir)

    ncols : int or None
        Number of columns in the grid. If None, chosen automatically.

    figsize : tuple
        Figure size

    norm : astropy norm, optional
        Custom normalization

    stretch : str
        Stretch for simple_norm (if norm is None)

    percent : float
        Percentile for normalization

    cmap : str
        Colormap

    save_path : str, optional
        Path to save the figure
    """

    # Get file list
    if psf_files is None:
        if psf_dir is None:
            raise ValueError("Provide either 'psf_dir' or 'psf_files'")
        psf_files = sorted(glob.glob(os.path.join(psf_dir, "*.fits")))

    n_psf = len(psf_files)

    if n_psf == 0:
        raise ValueError("No PSF files found")

    # Figure
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n_psf)))

    nrows = int(np.ceil(n_psf / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, psf_file in zip(axes, psf_files):
        with fits.open(psf_file) as hdul:
            ext = find_ext(hdul)
            if ext is None:
                raise ValueError(f"No valid image data in '{psf_file}'")

            psf_data = hdul[ext].data

        try:
            filt_name = get_filter(psf_file)
        except Exception:
            filt_name = os.path.basename(psf_file).replace(".fits", "")

        # Normalization
        if norm is None:
            norm_used = simple_norm(psf_data, stretch=stretch, percent=percent)
        else:
            norm_used = norm

        im = ax.imshow(psf_data, norm=norm_used, origin="lower", cmap=cmap)

        ax.set_title(filt_name, fontsize=10)
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")

        ny, nx = psf_data.shape
        ax.set_xticks([0, nx//2, nx-1])
        ax.set_yticks([0, ny//2, ny-1])

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Remove empty panels
    for j in range(n_psf, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Save
    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()

