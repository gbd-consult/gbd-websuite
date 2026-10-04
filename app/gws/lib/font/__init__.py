"""Fonts for server-side rendering.

This package installs custom fonts on the server and loads fonts for drawing.
Fonts are configured in the ``fonts`` option of the application. On configuration,
all files from the given directory are copied to the system font directory
``/usr/local/share/fonts`` and the font cache is rebuilt with ``fc-cache``.
Installed fonts can then be loaded by name with ``from_name``, for example, to draw SVG labels.

Example::

    fonts {
        dir "/data/fonts"
    }
"""

import PIL.ImageFont

import gws
import gws.lib.osx


class Config(gws.Config):
    """Custom fonts, installed on the server for rendering."""

    dir: gws.DirPath
    """Directory with font files to install."""


def configure(cfg: Config):
    """Install fonts according to the configuration.

    Args:
        cfg: Font configuration. If ``dir`` is not set, nothing is done.
    """
    if cfg.dir:
        install_fonts(cfg.dir)


def install_fonts(source_dir):
    """Copy font files to the system font directory and rebuild the font cache.

    Args:
        source_dir: Directory with font files. Files in subdirectories are also copied.
    """
    target_dir = '/usr/local/share/fonts'
    gws.lib.osx.run(['mkdir', '-p', target_dir], echo=True)
    for p in gws.lib.osx.find_files(source_dir):
        gws.lib.osx.run(['cp', '-v', p, target_dir], echo=True)

    gws.lib.osx.run(['fc-cache', '-fv'], echo=True)


def from_name(name: str, size: int):
    """Load a TrueType or OpenType font.

    Args:
        name: Font file name or path, as understood by ``PIL.ImageFont.truetype``.
        size: Font size in points.

    Returns:
        A ``PIL.ImageFont.FreeTypeFont`` object.
    """
    return PIL.ImageFont.truetype(font=name, size=size)
