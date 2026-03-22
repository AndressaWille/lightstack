"""
Lightstack: Tools for processing, PSF matching, and visualization
of multi-filter photometric datacubes using astronomical imaging (HST and JWST).

Main functionalities include:
- Crop regions
- Image alignment and photometric datacube construction
- PSF matching using convolution kernels
- Visualization of FITS images and datacubes
"""

# =========================
# Cropping 
# =========================
from .crop import (
    get_sky_bbox_from_cutout,
    crop_from_radec,
    crop_using_reference,
    crop_reg,
    cut_region_2d,
)


# =========================
# Datacube / alignment
# =========================
from .datacube import (
    align_reproject_fits,
    build_datacube,
    cut_region_datacube,
    build_valid_datacube,
    remove_filter,
    update_cube_header,
)

# =========================
# PSF matching
# =========================
from .psf import (
    build_kernel,
    save_kernel,
    apply_kernel,
    psf_match_datacube,
)

# =========================
# Visualization
# =========================
from .plot import (
    visualize_fits,
    plot_datacube_filters,
    plot_psf_grid,
)

# =========================
# Utils 
# =========================
from .utils import (
    find_ext,
    infer_filter,
)

# =========================
# Public API
# =========================
__all__ = [
    # crop
    "get_sky_bbox_from_cutout",
    "crop_from_radec",
    "crop_using_reference",
    "crop_reg",
    "cut_region_2d",
    
    # datacube
    "align_reproject_fits",
    "build_datacube",
    "cut_region_datacube",
    "build_valid_datacube",
    "remove_filter",
    "update_cube_header",

    # psf
    "build_kernel",
    "save_kernel",
    "apply_kernel",
    "psf_match_datacube",

    # plot
    "visualize_fits",
    "plot_datacube_filters",
    "plot_psf_grid",

    # utils
    "find_ext",
    "infer_filter",
    "pick_folder",
    "get_filter",
    "filter_id",
    "save_fits",
    "sort_fits",
    "MJy_sr_to_jy",
    "get_pixel_scale",
    
]
