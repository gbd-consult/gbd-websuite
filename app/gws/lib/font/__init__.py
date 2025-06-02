import PIL.ImageFont

import gws
import gws.lib.osx


class Config(gws.Config):
    """Custom fonts configuration."""

    dir: gws.DirPath
    """Directory with custom fonts."""


def configure(cfg: Config):
    if cfg.dir:
        install_fonts(cfg.dir)


def install_fonts(source_dir):
    target_dir = '/usr/local/share/fonts'
    gws.lib.osx.run(['mkdir', '-p', target_dir], echo=True)
    for p in gws.lib.osx.find_files(source_dir):
        gws.lib.osx.run(['cp', '-v', p, target_dir], echo=True)

    gws.lib.osx.run(['fc-cache', '-fv'], echo=True)


def from_name(name: str, size: int):
    return PIL.ImageFont.truetype(font=name, size=size)
