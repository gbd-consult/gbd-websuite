import gws
import gws.lib.metadata
from gws.lib.metadata import Record, Props


class Config(Record):
    """Metadata configuration"""
    pass


class Object(gws.Node, gws.IMetaData):
    _rec: Record

    def props_for(self, user):
        return Props(
            abstract=self._rec.abstract or '',
            attribution=self._rec.attribution or '',
            dateCreated=self._rec.dateCreated,
            dateUpdated=self._rec.dateUpdated,
            keywords=self._rec.keywords or [],
            language=self._rec.language or '',
            title=self._rec.title or '',
        )

    @property
    def values(self):
        return self._rec

    def configure(self):
        self._rec = gws.lib.metadata.from_dict(gws.as_dict(self.config))

    def extend(self, other):
        self._rec = gws.lib.metadata.merge(self._rec, other)

    def get(self, key):
        return gws.get(self._rec, key)

    def set(self, key, val):
        if '.' in key:
            ks = key.split('.')
            d = self._rec.get(ks[0])
        else:
            d = self._rec
        if d:
            d.set(key, val)
