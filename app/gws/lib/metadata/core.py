import gws
import gws.lib.intl

from . import inspire


class Props(gws.Props):
    """Represents metadata properties."""
    abstract: str
    attribution: str
    dateCreated: str
    dateUpdated: str
    keywords: list[str]
    language: str
    title: str


def from_dict(d: dict) -> gws.Metadata:
    """Creates a Metadata object from a dictionary.

    Args:
        d: Dictionary containing metadata information.

    Returns:
        A gws.Metadata object.
    """
    return check(gws.Metadata(d))


def from_args(**kwargs) -> gws.Metadata:
    """Creates a Metadata object from keyword arguments.

    Returns:
        A gws.Metadata object.
    """
    return from_dict(kwargs)


def from_config(c: gws.Config) -> gws.Metadata:
    """Creates a Metadata object from a configuration.

    Args:
        c: Configuration object.

    Returns:
        A gws.Metadata object.
    """
    return check(gws.Metadata(gws.u.to_dict(c)))


def from_props(p: gws.Props) -> gws.Metadata:
    """Creates a Metadata object from properties.

    Args:
        p: Properties object.

    Returns:
        A gws.Metadata object.
    """
    return check(gws.Metadata(gws.u.to_dict(p)))


def props(md: gws.Metadata) -> gws.Props:
    """Extracts properties from a Metadata object.

    Args:
        md: A gws.Metadata object.

    Returns:
        A gws.Props object containing metadata properties.
    """
    return gws.Props(
        abstract=md.abstract or '',
        attribution=md.attribution.title if md.attribution else '',
        dateCreated=md.dateCreated,
        dateUpdated=md.dateUpdated,
        keywords=md.keywords or [],
        language=md.language or '',
        title=md.title or '',
    )


_LIST_KEYS = [
    p
    for p, typ in gws.Metadata.__annotations__.items()
    if 'list' in repr(typ).lower()
]


def set_value(md: gws.Metadata, key: str, val) -> gws.Metadata:
    """Sets a value for a given key in a Metadata object.

    Args:
        md: A gws.Metadata object.
        key: The key to set.
        val: The value to set.

    Returns:
        The updated gws.Metadata object.
    """
    setattr(md, key, val)
    return check(md)


def set_default(md: gws.Metadata, key: str, val) -> gws.Metadata:
    """Sets a default value for a given key in a Metadata object if it doesn't exist.

    Args:
        md: A gws.Metadata object.
        key: The key to check.
        val: The default value to set.

    Returns:
        The updated gws.Metadata object.
    """
    if hasattr(md, key):
        return md
    setattr(md, key, val)
    return check(md)


def check(md: gws.Metadata) -> gws.Metadata:
    """Validates and normalizes a Metadata object.

    Args:
        md: A gws.Metadata object.

    Returns:
        A validated gws.Metadata object.
    """
    for key in _LIST_KEYS:
        val = md.get(key) or []
        if not isinstance(val, list):
            val = [val]
        setattr(md, key, val)

    p = md.language
    if p:
        md.language3 = gws.lib.intl.locale(p).language3
        md.languageBib = gws.lib.intl.locale(p).languageBib

    p = md.inspireTheme
    if p:
        md.inspireThemeName = inspire.theme_name(p, md.language)
        md.inspireThemeNameEn = inspire.theme_name(p, 'en')

    return md


def merge(*mds, extend_lists: bool = False) -> gws.Metadata:
    """Merges multiple Metadata objects.

    Args:
        *mds: Metadata objects to merge.
        extend_lists: Whether to extend list values instead of replacing them.

    Returns:
        A merged gws.Metadata object.
    """
    d = {}

    for md in mds:
        if not md:
            continue
        for key, val in gws.u.to_dict(md).items():
            if gws.u.is_empty(val):
                continue
            if extend_lists and key in _LIST_KEYS:
                val = d.get(key, []) + val
            d[key] = val

    return from_dict(d)
