"""Tools to deal with ini config files."""

import configparser
import io


def from_paths(*paths: str) -> dict:
    """Merges the key-value pairs of `.ini` files into a dictionary.

    Args:
        paths: Paths to `.ini` files.

    Returns:
        Nested dictionary with sections as keys and options as sub-keys.
    """

    res = {}
    cc = _from_paths(paths)

    for sec in cc.sections():
        for opt in cc.options(sec):
            res.setdefault(sec, {})[opt] = cc.get(sec, opt)

    return res


def from_paths_flat(*paths: str) -> dict:
    """Merges the key-value pairs of `.ini` files into a flat dictionary.

    Args:
        paths: Paths to `.ini` files.

    Returns:
        Flat dictionary with the section names as prefixes.
    """
    res = {}
    cc = _from_paths(paths)

    for sec in cc.sections():
        for opt in cc.options(sec):
            res[f'{sec}.{opt}'] = cc.get(sec, opt)

    return res


def _from_paths(paths):
    cc = configparser.ConfigParser()
    cc.optionxform = lambda optionstr: str(optionstr)

    for path in paths:
        cc.read(path)

    return cc


def to_string(d: dict) -> str:
    """Converts key-value pairs in a dictionary to a string grouped in sections.

    Args:
        d: Key-value pairs.

    Returns:
        String formatted like `.ini` files.
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
