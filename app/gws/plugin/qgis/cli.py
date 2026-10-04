"""Command-line commands for QGIS projects."""

import re

import gws
import gws.config
import gws.base.shape
import gws.lib.crs
import gws.lib.dynimport
import gws.lib.jsonx

from . import project


class CapsParams(gws.CliParams):
    """Parameters for the caps command."""

    src: str
    """Project file path or Postgres address ``postgres:<dbUid>/<schema>/<projectName>``."""
    out: str = ''
    """Output file name. If empty, the result is printed."""


class CopyParams(gws.CliParams):
    """Parameters for the copy command."""

    src: str
    """Source project file path or Postgres address ``postgres:<dbUid>/<schema>/<projectName>``."""
    dst: str
    """Destination project file path or Postgres address ``postgres:<dbUid>/<schema>/<projectName>``."""


class Object(gws.Node):
    """CLI commands for QGIS projects."""

    @gws.ext.command.cli('qgisCopy')
    def do_copy(self, p: CopyParams):
        """Copy a QGIS project between files and databases."""

        root = gws.config.load()
        src_prj = project.from_store(root, _addr_to_store(p.src))
        src_prj.to_store(root, _addr_to_store(p.dst))

    @gws.ext.command.cli('qgisCaps')
    def do_caps(self, p: CapsParams):
        """Print the parsed capabilities of a QGIS project as JSON."""

        root = gws.config.load()
        src_prj = project.from_store(root, _addr_to_store(p.src))
        caps = src_prj.caps()

        js = gws.lib.jsonx.to_pretty_string(caps, default=_caps_json)

        if p.out:
            gws.u.write_file(p.out, js)
            gws.log.info(f'saved to {p.out!r}')
        else:
            print(js)


def _addr_to_store(addr):
    m = re.match(r'^postgres:(.*?)/(.*?)/(.+?)$', addr)
    if m:
        return project.Store(type=project.StoreType.postgres, dbUid=m.group(1), schema=m.group(2), projectName=m.group(3))
    m = re.match(r'^(/.+)$', addr)
    if m:
        return project.Store(
            type=project.StoreType.file,
            path=m.group(1),
        )
    raise ValueError('invalid path or postgres address')


def _caps_json(x):
    if isinstance(x, gws.Crs):
        return x.epsg
    if isinstance(x, gws.base.shape.Shape):
        return x.to_geojson()
    try:
        return vars(x)
    except TypeError:
        return repr(x)
