import numpy as np
import os
import glob
import re

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


# Filter sets (JWST and HST)
NIRCAM = {"F070W","F090W","F115W","F140M","F150W","F162M","F164N","F187N","F182M","F200W","F210M",
          "F250M","F277W","F300M","F335M","F356W","F360M","F410M","F430M","F444W","F460M","F480M"}
ACS = {"F435W","F475W","F555W","F606W", "F625W","F775W","F814W","F850LP"}
WFC3_IR = {"F098M","F105W","F110W","F125W","F140W","F160W","F127M","F139M","F153M"}
WFC3_UV = {"F225W","F275W","F336W","F390W"}


def pick_folder(filt):
    """
    Returns the instrument folder name based on the filter.
    """
    if filt in NIRCAM:
        return "NIRCam"
    if filt in ACS:
        return "ACS"
    if filt in WFC3_IR:
        return "WFC3-IR"
    if filt in WFC3_UV:
        return "WFC3-UVIS"
    return "OTHERS"


def get_filter(fname):
    """
    Extracts the filter name from a FITS filename using a regular expression.

    Parameters
    ----------
    fname : str
        FITS filename.

    Returns
    -------
    filt : str
        Filter identifier (e.g., 'F150W', 'F410M').

    Raises
    ------
    ValueError
        If no filter name is found in the filename.
    """
    m = re.search(r'F\d{3}[WNM]', fname.upper())
    if m:
        return m.group(0)
    raise ValueError(f"Filter not found in filename: {fname}")


def infer_filter(fname):
    """
    Infers the filter name from a FITS filename.

    Parameters
    ----------
    fname : str
        FITS filename.

    Returns
    -------
    filt : str
        Inferred filter name.
    """
    base = os.path.basename(fname).upper().replace('.', '_')
    for part in base.split('_'):
        if part.startswith('F') and any(c.isdigit() for c in part):
            return part
    return os.path.splitext(os.path.basename(fname))[0]


def filter_id(filt):
    """
    Extracts the numeric part of a filter name for sorting.

    Parameters
    ----------
    filt : str
        Filter name.

    Returns
    -------
    num : int
        Numeric wavelength identifier. If not found, returns a large value
        so the filter is sorted last.
    """
    m = re.search(r'F(\d+)', filt)
    return int(m.group(1)) if m else 9999



def find_ext(hdul):
    """
    Finds the first FITS extension containing valid image data.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Opened FITS file.

    Returns
    -------
    ext : int or None
        Index of the HDU containing image data. Returns None if not found.
    """
    for i, hdu in enumerate(hdul):
        if hdu.data is not None and isinstance(hdu.data, np.ndarray):
            return i
    return None
    
    
def save_fits(data, header, path):
    """
    Saves a FITS file to disk.

    Parameters
    ----------
    data : numpy.ndarray
        Image data.
    header : astropy.io.fits.Header
        FITS header.
    path : str
        Output FITS path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)


def sort_fits(folder):
    """
    Reads all FITS files in a folder and sorts them by filter wavelength.

    Parameters
    ----------
    folder : str
        Path to the folder containing FITS files.

    Returns
    -------
    fits_sorted : list of tuples
        List in the form [(fits_path, filter_name), ...], sorted by filter id.
    """
    fits_files = glob.glob(os.path.join(folder, '*.fits'))
    fits_list = [(f, infer_filter(f)) for f in fits_files]
    fits_sorted = sorted(fits_list, key=lambda x: filter_id(x[1]))
    return fits_sorted

def MJy_sr_to_jy(aligned_list):
    """
    Convert FITS images from MJy/sr to Jy/pixel using the PIXAR_SR keyword: Jy/pixel = (MJy/sr) * 1e6 * PIXAR_SR

    Parameters
    ----------
    aligned_list : list of tuples
        List in the form [(fits_path, filter_name), ...].

    Returns
    -------
    new_list : list of tuples
        List in the form [(new_fits_path, filter_name), ...] for the converted files.

    """
    new_list = []

    for fpath, filt in aligned_list:

        with fits.open(fpath) as hdul:
            ext = find_ext(hdul)
            if ext is None:
                print(f"No data extension found in '{fpath}'. Skipping.")
                continue

            data = hdul[ext].data.astype(np.float64)
            header = hdul[ext].header.copy()

            # Pixel area in steradians
            pixar_sr = header.get('PIXAR_SR')
            if pixar_sr is None:
                print(f"PIXAR_SR not found in '{fpath}'. Skipping.")
                continue

            # Conversion factor: MJy to Jy and multiply by pixel area
            factor = 1e6 * pixar_sr
            data_jy = data * factor

            # Update header
            header['BUNIT'] = 'Jy'
            header.add_history(
                f"Converted from MJy/sr to Jy/pixel using PIXAR_SR = {pixar_sr}"
            )

            # Output file name
            out_name = os.path.splitext(fpath)[0] + "_Jy.fits"

            # Save
            fits.PrimaryHDU(data_jy, header=header).writeto(out_name, overwrite=True)
            print(f"Saved: {out_name}")

            new_list.append((out_name, filt))

    return new_list

def get_pixel_scale(fits_path):
    """
    Compute pixel scale in arcsec/pixel using WCS. Assumes square pixels and no significant distortion.
    """
    with fits.open(fits_path) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No valid image data in '{fits_path}'.")

        wcs = WCS(hdul[ext].header)

    pixscale = proj_plane_pixel_scales(wcs)[0] * 3600.0
    return pixscale
    
    
def get_pixel_scale_from_wcs(wcs):
    """
    Compute pixel scale in arcsec/pixel from a WCS object.
    Assumes square pixels and no significant distortion.
    """
    return proj_plane_pixel_scales(wcs)[0] * 3600.0
