"""Common xml writer helper."""

import gws
import gws.types as t

from . import namespaces


class NamespaceConfig(t.Data):
    name: str
    uri: t.Url
    schemaLocation: t.Optional[t.Url]


class Config(t.WithType):
    """XML settings"""

    namespaces: t.Optional[t.List[NamespaceConfig]]


class Object(gws.Object):
    def configure(self):
        super().configure()

        self.namespaces = {}

        for ns in namespaces.ALL:
            self.namespaces[ns[0]] = ns[1:]

        for ns in self.var('namespaces', default=[]):
            self.namespaces[ns.name] = [ns.uri, ns.get('schemaLocation', '')]
