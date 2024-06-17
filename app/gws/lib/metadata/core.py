import gws
import gws.lib.intl

from . import inspire


class Props(gws.Props):
    abstract: str
    attribution: str
    dateCreated: str
    dateUpdated: str
    keywords: list[str]
    language: str
    title: str


def from_dict(d: dict) -> gws.Metadata:
    return check(gws.Metadata(d))


def from_args(**kwargs) -> gws.Metadata:
    return from_dict(kwargs)


def from_config(c: gws.Config) -> gws.Metadata:
    return check(gws.Metadata(gws.u.to_dict(c)))


def from_props(p: gws.Props) -> gws.Metadata:
    return check(gws.Metadata(gws.u.to_dict(p)))


def props(md: gws.Metadata) -> gws.Props:
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
    setattr(md, key, val)
    return check(md)


def set_default(md: gws.Metadata, key: str, val) -> gws.Metadata:
    if hasattr(md, key):
        return md
    setattr(md, key, val)
    return check(md)


def check(md: gws.Metadata) -> gws.Metadata:
    for key in _LIST_KEYS:
        val = md.get(key) or []
        if not isinstance(val, list):
            val = [val]
        setattr(md, key, val)

    p = md.language
    if p:
        md.language3 = gws.lib.intl.locale(p).language3

    p = md.inspireTheme
    if p:
        md.inspireThemeName = inspire.theme_name(p, md.language)
        md.inspireThemeNameEn = inspire.theme_name(p, 'en')

    return md


def merge(*mds, extend_lists=False) -> gws.Metadata:
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
