import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.units as u

from .utils import find_ext

def get_sky_bbox_from_cutout(fits_cut):
    """
    Get sky bounding box (RA, Dec) from a FITS cutout.

    Parameters
    ----------
    fits_cut : str
        Path to FITS file

    Returns
    -------
    ra, dec : arrays
        RA and Dec of the four corners
    """
    # Read FITS
    with fits.open(fits_cut) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No 2D image found in {fits_cut}")

        data = hdul[ext].data
        header = hdul[ext].header

        wcs = WCS(header)

    ny, nx = data.shape

    # Define pixel coordinates of the four corners
    corners_pix = np.array([
        [0, 0],
        [nx-1, 0],
        [0, ny-1],
        [nx-1, ny-1]])

    # Convert pixel coordinates to sky (RA, Dec)
    ra, dec = wcs.wcs_pix2world(
        corners_pix[:, 0],
        corners_pix[:, 1],
        0  # origin = 0 (Python convention)
    )

    return ra, dec


def crop_from_radec(fits_path, ra, dec, size_arcsec, output_path=None):
    """
    Extract a square cutout from a FITS image centered on a given sky position.

    Parameters
    ----------
    fits_path : str
        Path to the FITS image.

    ra : float
        Right Ascension of the center (in degrees).

    dec : float
        Declination of the center (in degrees).

    size_arcsec : float
        Size of the cutout (in arcseconds). The cutout is square.

    Returns
    -------
    data_cut : 2D numpy array
        Cropped image data.

    header_cut : FITS header
        Updated header with corrected WCS reference (CRPIX).
    """

    # Open FITS
    with fits.open(fits_path) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No 2D image data found in {fits_path}")

        data = hdul[ext].data
        header = hdul[ext].header
        wcs = WCS(header)

    # Convert sky coordinates (RA, Dec) to pixel coordinates
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    x_center, y_center = skycoord_to_pixel(coord, wcs)

    # Get pixel scale (arcsec/pixel)
    pixel_scales = proj_plane_pixel_scales(wcs) * 3600.0  # deg → arcsec

    pixscale = np.mean(pixel_scales)

    # Convert size from arcsec to pixels
    half_size_pix = (size_arcsec / pixscale) / 2.0

    # Define integer pixel bounds
    x_min = int(np.floor(x_center - half_size_pix))
    x_max = int(np.ceil(x_center + half_size_pix))
    y_min = int(np.floor(y_center - half_size_pix))
    y_max = int(np.ceil(y_center + half_size_pix))

    ny, nx = data.shape
    x_min = max(0, x_min)
    x_max = min(nx, x_max)
    y_min = max(0, y_min)
    y_max = min(ny, y_max)

    # Extract cutout
    data_cut = data[y_min:y_max, x_min:x_max]

    # Update WCS
    header_cut = header.copy()
    header_cut['CRPIX1'] -= x_min
    header_cut['CRPIX2'] -= y_min

    if output_path is not None:
        fits.PrimaryHDU(data_cut, header=header_cut).writeto(
        output_path,
        overwrite=True)
    
    return data_cut, header_cut


def crop_using_reference(fits_path, ref_cutout, output_path=None):
    """
    Crop a FITS image using the sky footprint of a reference cutout.

    This function extracts a region from a target image such that it matches
    the sky coverage (RA/Dec bounding box) of a reference FITS cutout.

    Parameters
    ----------
    fits_path : str
        Path to the target FITS image to be cropped.

    ref_cutout : str
        Path to the reference FITS cutout defining the sky region.

    output_path : str, optional
        If provided, saves the cropped FITS to this path.

    Returns
    -------
    data_cut : 2D numpy array
        Cropped image data.

    header_cut : FITS header
        Updated header with corrected WCS reference (CRPIX).
    """

    # Get RA/Dec bounding box from reference cutout
    ra, dec = get_sky_bbox_from_cutout(ref_cutout)

    # Open FITS
    with fits.open(fits_path) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No 2D image found in {fits_path}")

        data = hdul[ext].data
        header = hdul[ext].header
        wcs = WCS(header)

    # Convert sky coordinates (RA/Dec) to pixel coordinates 
    x, y = wcs.wcs_world2pix(ra, dec, 0)

    # Define bounding box in pixel space
    x_min = int(np.floor(np.min(x)))
    x_max = int(np.ceil(np.max(x)))
    y_min = int(np.floor(np.min(y)))
    y_max = int(np.ceil(np.max(y)))

    ny, nx = data.shape
    x_min = max(0, x_min)
    x_max = min(nx, x_max)
    y_min = max(0, y_min)
    y_max = min(ny, y_max)

    # Extract cutout
    data_cut = data[y_min:y_max, x_min:x_max]

    # Update WCS
    header_cut = header.copy()
    header_cut['CRPIX1'] -= x_min
    header_cut['CRPIX2'] -= y_min

    # Save 
    if output_path is not None:
        fits.PrimaryHDU(data_cut, header=header_cut).writeto(
            output_path,
            overwrite=True)

    return data_cut, header_cut
    

def crop_reg(fits_path, region):
    """
    Crops a FITS image using a DS9 region.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.
    region : regions.Region
        Region object read from a DS9 region file.

    Returns
    -------
    data_cut : numpy.ndarray
        Cropped image data.
    header_cut : astropy.io.fits.Header
        Updated FITS header with adjusted CRPIX keywords.

    Raises
    ------
    ValueError
        If no image data extension is found in the FITS file.
    """
    # Open fits
    with fits.open(fits_path) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No image data found in {fits_path}")

        data = hdul[ext].data
        header = hdul[ext].header
        wcs = WCS(header)
    
    # Define region limits
    pix_region = region.to_pixel(wcs)
    bbox = pix_region.bounding_box

    x_min, x_max = int(bbox.ixmin), int(bbox.ixmax)
    y_min, y_max = int(bbox.iymin), int(bbox.iymax)

    # Crop the image
    data_cut = data[y_min:y_max, x_min:x_max]

    # Update reference pixel in the header
    header_cut = header.copy()
    header_cut['CRPIX1'] -= x_min
    header_cut['CRPIX2'] -= y_min

    return data_cut, header_cut
    
    
def cut_region_2d(fits_file, x_start, x_end, y_start, y_end, output_path):
    """
    Cuts a spatial region from a 2D fits image.

    Parameters
    ----------
    fits_file : str
        Path to the input 2D fits image.
    x_start, x_end : int
        Pixel indices for the x axis.
    y_start, y_end : int
        Pixel indices for the y axis.
    output_path : str
        Path to the output fits file.

    Returns
    -------
    None
        Saves the cut fits image.
    """
    # Open FITS
    with fits.open(fits_file) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No image data found in {fits_path}")

        data = hdul[ext].data
        header = hdul[ext].header

        # Cut the data
        cut_data = data[y_start:y_end, x_start:x_end]

        # Update WCS
        wcs_2d = WCS(header)
        wcs_2d.wcs.crpix[0] -= x_start
        wcs_2d.wcs.crpix[1] -= y_start

        # Build new header
        new_header = wcs_2d.to_header()
        new_header['NAXIS'] = 2
        new_header['NAXIS1'] = x_end - x_start
        new_header['NAXIS2'] = y_end - y_start

        # Preserve unit if exists
        if 'BUNIT' in header:
            new_header['BUNIT'] = header['BUNIT']

        # Optional: store crop info
        new_header['XMINPIX'] = x_start
        new_header['XMAXPIX'] = x_end
        new_header['YMINPIX'] = y_start
        new_header['YMAXPIX'] = y_end

        # Save cut image
        fits.PrimaryHDU(cut_data, header=new_header).writeto(output_path, overwrite=True)
        print(f"Cut 2D fits saved to '{output_path}'")
