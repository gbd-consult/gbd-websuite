import gws
import gws.lib.intl
import gws.types as t

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
    return gws.Metadata(d)


def from_args(**kwargs) -> gws.Metadata:
    return from_dict(kwargs)


def from_config(config) -> gws.Metadata:
    return gws.Metadata(gws.to_dict(config))


def from_props(props: gws.Props) -> gws.Metadata:
    return gws.Metadata(gws.to_dict(props))


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
    if 'List' in repr(typ)
]


def set_value(md: gws.Metadata, key: str, val: t.Any) -> gws.Metadata:
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
        md.language3 = gws.lib.intl.bibliographic_name(language=p)

    p = md.inspireTheme
    if p:
        md.inspireThemeName = inspire.theme_name(p, md.language)
        md.inspireThemeNameEn = inspire.theme_name(p, 'en')

    return md


def extend(md: gws.Metadata, *others, extend_lists=False) -> gws.Metadata:
    for other in others:
        if not other:
            continue
        if hasattr(other, 'md'):
            other = getattr(other, 'md')

        for key, val in gws.to_dict(other).items():
            if gws.is_empty(val):
                continue
            if extend_lists and key in _LIST_KEYS:
                val = getattr(md, key, []) + val
            setattr(md, key, val)

    return check(md)
