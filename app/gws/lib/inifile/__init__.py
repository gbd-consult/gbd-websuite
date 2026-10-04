"""Read and write ini files.

Reads one or more ini files with ``configparser`` into a nested dict (section, then option)
or into a flat dict with ``section.option`` keys, and writes a flat dict back as ini text.
When reading, option names keep their case, files that do not exist are skipped, and values
from later files override earlier ones.

Example::

    d = gws.lib.inifile.from_paths_flat('/data/a.ini', '/data/b.ini')
    # {'section.option': 'value', ...}
    text = gws.lib.inifile.to_string({'section.option': 'value'})
"""

import configparser
import io


def from_paths(*paths: str) -> dict:
    """Read ini files into a nested dict.

    Args:
        *paths: Paths to ini files. Missing files are skipped, later files override earlier ones.

    Returns:
        A dict with section names as keys and dicts of options as values.
    """

    res = {}
    cc = _from_paths(paths)

    for sec in cc.sections():
        for opt in cc.options(sec):
            res.setdefault(sec, {})[opt] = cc.get(sec, opt)

    return res


def from_paths_flat(*paths: str) -> dict:
    """Read ini files into a flat dict.

    Args:
        *paths: Paths to ini files. Missing files are skipped, later files override earlier ones.

    Returns:
        A dict with ``section.option`` keys.
    """
    res = {}
    cc = _from_paths(paths)

    for sec in cc.sections():
        for opt in cc.options(sec):
            res[f'{sec}.{opt}'] = cc.get(sec, opt)

    return res


def _from_paths(paths):
    """Read ini files into a case-sensitive ``ConfigParser``."""
    cc = configparser.ConfigParser()
    cc.optionxform = lambda optionstr: str(optionstr)

    for path in paths:
        cc.read(path)

    return cc


def to_string(d: dict) -> str:
    """Convert a flat dict to ini text.

    Keys are split at the first dot into a section name and an option name.
    Option names are lowercased.

    Args:
        d: A dict with ``section.option`` keys and string values.

    Returns:
        Ini file content.
    """

    cc = configparser.ConfigParser()

    for k, v in d.items():
        sec, _, name = k.partition('.')
        if not cc.has_section(sec):
            cc.add_section(sec)
        cc.set(sec, name, v)

    with io.StringIO() as fp:
        cc.write(fp, space_around_delimiters=False)
        return fp.getvalue()
