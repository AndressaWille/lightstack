import numpy as np
import os

from astropy.io import fits
from astropy.wcs import WCS

from reproject import reproject_interp, reproject_exact

from .utils import find_ext, infer_filter


def align_reproject_fits(fits_list, ref_file, method="interp", crop=1):
    """
    Aligns and reprojects FITS images to a common WCS using a reference FITS file.

    Parameters
    ----------
    fits_list : list of tuples
        List in the form [(fits_path, filter_name), ...].

    ref_file : str
        Path to the FITS file used as WCS reference.

    method : str, optional
        Reprojection method:
        - "interp" (default): faster, interpolates values
        - "exact": slower, conserves flux   --> but reproject_exact has precision issues with resolutions below ~0.05 arcsec, so the results may not be accurate.

    crop : int, optional
        Number of pixels to remove from each border after reprojection
        (helps remove edge artifacts from interpolation).

    Returns
    -------
    aligned_list : list of tuples
        List in the form [(aligned_fits_path, filter_name), ...].
    """

    # Choose reprojection method
    if method == "exact":
        reproj_func = reproject_exact
    elif method == "interp":
        reproj_func = reproject_interp
    else:
        raise ValueError("method must be 'interp' or 'exact'")

    # Open reference FITS
    with fits.open(ref_file) as hdul_ref:
        ext_ref = find_ext(hdul_ref)
        if ext_ref is None:
            raise ValueError(f"No valid image data in reference file '{ref_file}'.")

        ref_header = hdul_ref[ext_ref].header
        ref_wcs = WCS(ref_header)
        shape_out = hdul_ref[ext_ref].data.shape

    aligned_list = []

    # Loop over all FITS
    for fpath, filt in fits_list:

        with fits.open(fpath) as hdul:
            ext = find_ext(hdul)
            if ext is None:
                print(f"No data extension found in {fpath}. Skipping.")
                continue

            data = hdul[ext].data
            header = hdul[ext].header
            wcs_in = WCS(header)

            unit = header.get('BUNIT', 'unknown')
            print(f"Filter {filt}: unit = {unit}")

            # Reproject
            data_aligned, footprint = reproj_func(
                (data, wcs_in),
                ref_wcs,
                shape_out=shape_out)

            # Crop border (interp method)
            if crop > 0:
                data_aligned = data_aligned[crop:-crop, crop:-crop]

                # Adjust WCS
                wcs_out = ref_wcs.deepcopy()
                wcs_out.wcs.crpix -= crop
            else:
                wcs_out = ref_wcs

            # Header
            header_aligned = wcs_out.to_header()

            ref_filter = infer_filter(ref_file)
            header_aligned.add_history(
                f"Reprojected to {ref_filter} using {method} method (reproject package)")

            if crop > 0:
                header_aligned.add_history(f"Cropped {crop} pixels from each border")

            # Save
            suffix = f"_aligned_{method}"
            out_name = os.path.splitext(fpath)[0] + f"{suffix}.fits"

            fits.PrimaryHDU(data_aligned, header=header_aligned).writeto(
                out_name, overwrite=True)

            aligned_list.append((out_name, filt))
            print(f"Saved: {out_name}")

    return aligned_list
    
    
def build_datacube(aligned_fits_files, reference_file, output_path):
    """
    Builds a 3D datacube from aligned 2D FITS images.

    Parameters
    ----------
    aligned_fits_files : list of tuples
        [(filename, filter_name), ...]
    reference_file : str
        FITS file to define WCS and shape.
    output_path : str
        Path to save the 3D datacube.
    """
    
    # Open FITS
    with fits.open(reference_file) as hdul_ref:
        ext_ref = find_ext(hdul_ref)
        if ext_ref is None:
            raise ValueError(f"No 2D data in reference file {reference_file}.")
        ref_header = hdul_ref[ext_ref].header
        ny, nx = hdul_ref[ext_ref].data.shape
        wcs_2d = WCS(ref_header, naxis=2)

    cube_images = []
    filter_names = []
    units = []

    # Construct cube
    for file, filt in aligned_fits_files:
        with fits.open(file) as hdul:
            ext = find_ext(hdul)
            if ext is None:
                print(f"No 2D data in {file}. Skipping.")
                continue
            data = hdul[ext].data
            cube_images.append(data)
            filter_names.append(filt)
            units.append(hdul[ext].header.get('BUNIT', 'unknown'))

    cube = np.array(cube_images)

    # Write header
    wcs_3d = WCS(naxis=3)
    wcs_3d.wcs.crpix[0] = wcs_2d.wcs.crpix[0]
    wcs_3d.wcs.crpix[1] = wcs_2d.wcs.crpix[1]
    wcs_3d.wcs.crval[0] = wcs_2d.wcs.crval[0]
    wcs_3d.wcs.crval[1] = wcs_2d.wcs.crval[1]
    wcs_3d.wcs.cdelt[0] = wcs_2d.wcs.cdelt[0]
    wcs_3d.wcs.cdelt[1] = wcs_2d.wcs.cdelt[1]
    wcs_3d.wcs.ctype[0] = wcs_2d.wcs.ctype[0]
    wcs_3d.wcs.ctype[1] = wcs_2d.wcs.ctype[1]
    wcs_3d.wcs.cunit[0] = wcs_2d.wcs.cunit[0]
    wcs_3d.wcs.cunit[1] = wcs_2d.wcs.cunit[1]

    if wcs_2d.wcs.has_cd():
        cd3 = np.zeros((3,3))
        cd3[0:2,0:2] = wcs_2d.wcs.cd.copy()
        cd3[2,2] = 1.0
        wcs_3d.wcs.cd = cd3

    wcs_3d.wcs.crpix[2] = 1
    wcs_3d.wcs.crval[2] = 0
    wcs_3d.wcs.cdelt[2] = 1
    wcs_3d.wcs.ctype[2] = 'FILTER'
    wcs_3d.wcs.cunit[2] = ''

    header = wcs_3d.to_header()
    header['NAXIS'] = 3
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['NAXIS3'] = len(filter_names)
    
    with fits.open(aligned_fits_files[0][0]) as hdul0:
        ext0 = find_ext(hdul0)
        header_meta = hdul0[ext0].header

    if "HISTORY" in header_meta:
        for h in header_meta["HISTORY"]:
            header.add_history(h)

    if all(u == units[0] for u in units):
        header['BUNIT'] = units[0]
    else:
        header['BUNIT'] = 'unknown'

    for i, filt in enumerate(filter_names):
        header[f'FILTER{i+1}'] = filt
        
    header['NFILTERS'] = len(filter_names)
    header['FILTERS'] = ",".join(filter_names)

    # Save datacube
    fits.PrimaryHDU(cube, header=header).writeto(output_path, overwrite=True)
    print(f"Datacube saved at {output_path}.")


def cut_region_datacube(cube_fits_file, x_start, x_end, y_start, y_end, output_path):
    """
    Cuts a spatial region from a datacube.

    Parameters
    ----------
    cube_fits_file : str
        Path to the input 3D fits datacube.
    x_start, x_end : int
        Pixel indices for the x axis.
    y_start, y_end : int
        Pixel indices for the y axis.
    output_filename : str
        Path to the output fits file.

    Returns
    -------
    None
        Saves the cut datacube.
    """
    # Open datacube
    with fits.open(cube_fits_file) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No image data found in {fits_path}")

        cube_data = hdul[ext].data
        cube_header = hdul[ext].header

        # Cut the data array
        cut_data = cube_data[:, y_start:y_end, x_start:x_end]

        # Get and update WCS
        wcs_3d = WCS(cube_header)
        wcs_3d.wcs.crpix[0] -= x_start
        wcs_3d.wcs.crpix[1] -= y_start

        # Create new header with updated size and WCS
        new_header = wcs_3d.to_header()
        new_header['NAXIS'] = 3
        new_header['NAXIS1'] = x_end - x_start
        new_header['NAXIS2'] = y_end - y_start
        new_header['NAXIS3'] = cube_data.shape[0]

        # Filter info
        for key in cube_header:
            if key.startswith('FILTER'):
                new_header[key] = cube_header[key]

        # Crop window
        new_header['XMINPIX'] = x_start
        new_header['XMAXPIX'] = x_end
        new_header['YMINPIX'] = y_start
        new_header['YMAXPIX'] = y_end

        # Save cut datacube
        fits.PrimaryHDU(cut_data, header=new_header).writeto(output_path, overwrite=True)
        print(f"Cut datacube saved to '{output_path}'")
        
def build_valid_datacube(cube_fits_file, output_cube, threshold=0.0, frac_valid=0.01):
    """
    Remove empty filters in a datacube, saves the new datacube and returns the valid filter names.
    
    Parameters
    ----------
    cube_fits_file : str
        Path to the input 3D fits datacube.
    output_cube: str
        Path to save the filtered fits. 
    threshold : float
        Minimum flux value to consider a pixel valid.
    frac_valid : float
        Minimum fraction of pixels above the threshold to consider the filter valid.
    
    Retorna
    -------
    cube_filtered : np.ndarray
        Datacube with only valid filters. 
    filters_valid : list
        List of valid filter names.
        
    Saves the new datacube.
    """
    
    # Open FITS
    with fits.open(cube_fits_file) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No image data found in {fits_path}")

        cube = hdul[ext].data
        header = hdul[ext].header
        
        n_filters = header.get("NAXIS3", cube.shape[0])
        filters = [header[f"FILTER{i+1}"].strip() for i in range(n_filters)]
    
    # Keep only valid filters
    valid_indices = []
    for i, img in enumerate(cube):
        valid = np.isfinite(img)
        n_total = valid.sum()
        if n_total == 0:
            continue
        n_above = np.sum(img[valid] > threshold)
        if (n_above / n_total) >= frac_valid:
            valid_indices.append(i)

    cube_filtered = cube[valid_indices]
    filters_valid = [filters[i] for i in valid_indices]
    
    # Write new header
    new_header = header.copy()
    new_header["NAXIS3"] = len(filters_valid)

    for i, f in enumerate(filters_valid):
        new_header[f"FILTER{i+1}"] = f

    for i in range(len(filters_valid), n_filters):
        key = f"FILTER{i+1}"
        if key in new_header:
            del new_header[key]

    # Save filtered datacube 
    hdu = fits.PrimaryHDU(cube_filtered, header=new_header)
    hdu.writeto(output_cube, overwrite=True)
    print(f"Filtered datacube saved at '{output_cube}'")

    return cube_filtered, filters_valid


def remove_filter(cube_fits_file, output_cube, filter_to_remove):
    """
    Removes a specific filter from a 3D datacube.

    Parameters
    ----------
    cube_fits_file : str
        Path to the original datacube.
    output_cube : str
        Path to save the new datacube.
    filter_to_remove : str
        Name of the filter to be removed (e.g., ‘F115W’).
    """

    # Open FITS
    with fits.open(cube_fits_file) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No image data found in {fits_path}")

        cube = hdul[ext].data
        header = hdul[ext].header

        n_filters = header.get("NAXIS3", cube.shape[0])
        filters = [header[f"FILTER{i+1}"].strip() for i in range(n_filters)]

    # Keep filters
    keep_indices = [i for i, f in enumerate(filters) if f != filter_to_remove]

    if len(keep_indices) == len(filters):
        print(f"This filter '{filter_to_remove}' is not in the datacube.")
    else:
        print(f"Removing filter: {filter_to_remove}")

    # New cube
    cube_new = cube[keep_indices]
    filters_new = [filters[i] for i in keep_indices]

    # Write new header
    new_header = header.copy()
    new_header["NAXIS3"] = len(filters_new)

    for i, f in enumerate(filters_new):
        new_header[f"FILTER{i+1}"] = f

    for i in range(len(filters_new), n_filters):
        key = f"FILTER{i+1}"
        if key in new_header:
            del new_header[key]

    # Save
    hdu = fits.PrimaryHDU(cube_new, header=new_header)
    hdu.writeto(output_cube, overwrite=True)
    print(f"New datacube without '{filter_to_remove}' saved at {output_cube}")
    
    return cube_new, filters_new
    
def update_cube_header(
    cube_fits_file,
    output_file=None,
    redshift=None,
    ra_center=None,
    dec_center=None,
    use_brightest_pixel=False,
    overwrite=False):
    """
    Update a datacube header with astrophysical metadata.

    By default, creates a new file with suffix '_more.fits'
    instead of overwriting the original.

    Parameters
    ----------
    cube_fits_file : str
        Path to input datacube.

    output_file : str or None
        Output FITS file. If None, creates '<original>_more.fits'.

    redshift : float or None
        Galaxy redshift.

    ra_center, dec_center : float or None
        Galaxy center coordinates (deg).

    use_brightest_pixel : bool
        If True, estimate center from brightest pixel.

    overwrite : bool
        Overwrite output file if it already exists.
    """

    # Define output file 
    if output_file is None:
        base, ext = os.path.splitext(cube_fits_file)
        output_file = f"{base}_more{ext}"

    # Open FITS
    with fits.open(cube_fits_file) as hdul:
        ext = find_ext(hdul)
        if ext is None:
            raise ValueError(f"No valid data extension in {cube_fits_file}")

        datacube = hdul[ext].data
        header = hdul[ext].header.copy()

        wcs = WCS(header)
        if wcs.pixel_n_dim > 2:
            wcs = wcs.celestial

    # Redshift
    if redshift is not None and not np.isnan(redshift):
        header["REDSHIFT"] = (redshift, "Galaxy redshift")
    else:
        header["REDSHIFT"] = ("UNKNOWN", "Galaxy redshift not provided")

    # Galaxy center
    if use_brightest_pixel:
        collapsed = np.nansum(datacube, axis=0)

        y_max, x_max = np.unravel_index(
            np.nanargmax(collapsed), collapsed.shape)

        ra_peak, dec_peak = wcs.wcs_pix2world(x_max, y_max, 0)

        header["GAL_XCEN"] = (x_max, "Galaxy center X (pixel)")
        header["GAL_YCEN"] = (y_max, "Galaxy center Y (pixel)")
        header["GAL_RA"] = (ra_peak, "Galaxy center RA (deg)")
        header["GAL_DEC"] = (dec_peak, "Galaxy center Dec (deg)")
        header["CEN_TYPE"] = ("BRIGHTEST", "Center definition")

    elif (ra_center is not None) and (dec_center is not None):
        header["GAL_RA"] = (ra_center, "Galaxy center RA (deg)")
        header["GAL_DEC"] = (dec_center, "Galaxy center Dec (deg)")
        header["CEN_TYPE"] = ("USER", "Center definition")

    else:
        header["CEN_TYPE"] = ("UNKNOWN", "Center not defined")

    # Save
    with fits.open(cube_fits_file) as hdul:
        hdul[ext].header = header
        hdul.writeto(output_file, overwrite=overwrite)

    print(f"Updated header saved to: {output_file}")
