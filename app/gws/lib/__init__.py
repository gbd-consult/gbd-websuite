"""General purpose libraries.

This package holds the libraries that the rest of GBD WebSuite builds on.
They wrap third-party packages and implement common formats and algorithms,
and are meant to be independent of the configurable objects in ``gws.base``:
in the intended import hierarchy, ``lib`` imports only from ``gws.core`` and
from other ``lib`` packages. Some libraries declare types in a ``types.pyinc`` file,
which are exported as ``gws.*`` types (for example ``gws.Crs``,
``gws.Image``, ``gws.MapGrid``, ``gws.XmlElement``).

Subpackages:

Geometry and coordinates

- ``bounds``: bounds (an extent with a CRS) utilities.
- ``crs``: coordinate reference systems and transformations.
- ``extent``: extent utilities.
- ``gdalx``: GDAL/OGR wrapper for raster and vector files.
- ``gml``: reading and writing GML geometries.
- ``grid``: map grids and tile pyramid math.
- ``uom``: units of measure and conversions.

Rendering and formats

- ``font``: fonts for server-side rendering.
- ``image``: raster images.
- ``mapserver``: MapServer support.
- ``pdf``: PDF utilities.
- ``style``: feature styles.
- ``svg``: SVG drawing and sanitizing.

Data formats and text

- ``cql``: CQL2 support.
- ``datetimex``: date and time utilities.
- ``htmlx``: HTML utilities.
- ``inifile``: reading and writing ini files.
- ``intl``: locales and locale-aware formatting.
- ``jsonx``: JSON utilities.
- ``mime``: MIME types.
- ``xmlx``: XML parsing, building and serialization.
- ``zipx``: zip archive utilities.

System, network and security

- ``cli``: utilities for command line scripts.
- ``dynimport``: dynamic imports of Python code.
- ``net``: URL utilities and an HTTP client.
- ``osx``: operating system and shell utilities.
- ``otp``: HOTP and TOTP one-time passwords.
- ``password``: password hashing, checking and generation.
- ``sa``: convenience wrapper for SQLAlchemy imports.
- ``sqlitex``: convenience wrapper for the SQLite driver.
- ``watcher``: file system watcher.

Third-party code

- ``vendor``: vendored third-party packages.

Example::

    import gws.lib.crs
    import gws.lib.extent
    import gws.lib.net
    import gws.lib.xmlx

    crs = gws.lib.crs.get(25832)
    ext = gws.lib.extent.transform(
        (500000, 5700000, 510000, 5710000), crs, gws.lib.crs.WEBMERCATOR)

    res = gws.lib.net.http_request('https://example.com/wms?request=GetCapabilities')
    root = gws.lib.xmlx.from_string(res.text)
"""
