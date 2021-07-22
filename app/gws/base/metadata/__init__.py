import gws
import gws.lib.metadata
from gws.lib.metadata import Values
import gws.types as t


class Config(Values):
    """Metadata configuration"""
    pass


class Props(gws.Props):
    abstract: str
    attribution: str
    dateCreated: str
    dateUpdated: str
    keywords: t.List[str]
    language: str
    title: str


class Object(gws.Node, gws.IMetaData):
    _values: Values

    @property
    def props(self):
        return gws.Props(
            abstract=self._values.abstract or '',
            attribution=self._values.attribution or '',
            dateCreated=self._values.dateCreated,
            dateUpdated=self._values.dateUpdated,
            keywords=self._values.keywords or [],
            language=self._values.language or '',
            title=self._values.title or '',
        )

    @property
    def values(self) -> Values:
        return self._values

    def configure(self):
        self._values = gws.lib.metadata.from_dict(gws.as_dict(self.config))

    def extend(self, other):
        self._values = gws.lib.metadata.merge(self._values, other)

    def get(self, key):
        return gws.get(self._values, key)

    def set(self, key, val):
        if '.' in key:
            ks = key.split('.')
            d = self._values.get(ks[0])
        else:
            d = self._values
        if d:
            d.set(key, val)
