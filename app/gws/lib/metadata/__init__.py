"""Metadata structures and related utilities"""

import gws
import gws.lib.country
import gws.types as t

from . import inspire


##

class Props(gws.Props):
    abstract: str
    attribution: str
    dateCreated: str
    dateUpdated: str
    keywords: t.List[str]
    language: str
    title: str


class ExtendOption(t.Enum):
    app = 'app'  #: substutute missing metadata from the project or application config
    source = 'source'  #: substutute missing metadata from the source


class Config(gws.MetadataValues):
    """Metadata configuration"""

    extend: t.Optional[ExtendOption]


##

def from_dict(d: dict) -> 'Metadata':
    return Metadata(d)


def from_args(**kwargs) -> 'Metadata':
    return from_dict(kwargs)


def from_config(cfg: gws.Config) -> 'Metadata':
    return from_dict(gws.to_dict(cfg))


def from_props(props: gws.Props) -> 'Metadata':
    return from_dict(gws.to_dict(props))


##


_LIST_PROPS = {'metaLinks', 'extraLinks'}
_EXT_LIST_PROPS = {'keywords', 'inspireKeywords'}
_NO_EXT_PROPS = {'authorityIdentifier', 'catalogUid'}


class Metadata(gws.Object, gws.IMetadata):
    def __init__(self, d):
        super().__init__()
        self._update(d)

    def props_for(self, user):
        return gws.Data(
            abstract=self.values.abstract or '',
            attribution=self.values.attribution or '',
            dateCreated=self.values.dateCreated,
            dateUpdated=self.values.dateUpdated,
            keywords=self.values.keywords or [],
            language=self.values.language or '',
            title=self.values.title or '',
        )

    def extend(self, *others):
        d = gws.to_dict(self.values)

        for o in others:
            if not o:
                continue

            if isinstance(o, Metadata):
                o = o.values

            for k, new in gws.to_dict(o).items():
                if k in _NO_EXT_PROPS or gws.is_empty(new):
                    continue
                old = d.get(k)
                if old is None:
                    d[k] = new
                elif k in _EXT_LIST_PROPS:
                    old.extend(new)

        self._update(d)
        return self

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, val):
        d = gws.to_dict(self.values)
        d[key] = val
        self._update(d)
        return self

    def _update(self, d):
        for k in _EXT_LIST_PROPS:
            d[k] = sorted(set(d.get(k, [])))
        for k in _LIST_PROPS:
            d[k] = d.get(k, [])

        if d.get('language'):
            d['language3'] = gws.lib.country.bibliographic_name(language=d['language'])

        if d.get('inspireTheme'):
            d['inspireThemeName'] = inspire.theme_name(d['inspireTheme'], d.get('language'))
            d['inspireThemeNameEn'] = inspire.theme_name(d['inspireTheme'], 'en')

        self.values = gws.MetadataValues(d)
