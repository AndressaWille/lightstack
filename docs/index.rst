.. lightstack documentation master file, created by
   sphinx-quickstart on Mon Mar 23 12:13:55 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lightstack
======================================

A Python package for building photometric data cubes from multi-band imaging.


Photometric data
----------------

Photometric data has become essential in modern astrophysics. These multi-band observations allow us to probe different physical processes across a wide range of wavelengths.

This package provides tools to build photometric data cubes by organizing filters in wavelength space and stacking them consistently. Although Lightstack was developed and tested using data from the James Webb Space Telescope (JWST) and the Hubble Space Telescope (HST), the methodology may be applicable to any multi-band photometric dataset (with imaging data in FITS format with WCS information).


Install
-------

From PyPI
~~~~~~~~~

.. code-block:: bash

    pip install lightstack

Source code
~~~~~~~~~~~

https://github.com/AndressaWille/lightstack

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   workflow
   examples
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
