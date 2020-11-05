"""Common xml writer helper."""

import gws
import gws.types as t

from . import namespaces


class NamespaceConfig(t.Data):
    """XML namespace configuration"""

    name: str  #: namespace name
    uri: t.Url  #: namespace uri
    schemaLocation: t.Optional[t.Url]  #: namespace schema location


class Config(t.WithType):
    """XML settings"""

    namespaces: t.Optional[t.List[NamespaceConfig]]  #: custom namespaces


class Object(gws.Object):
    def configure(self):
        super().configure()

        self.namespaces = {}

        for ns in namespaces.ALL:
            self.namespaces[ns[0]] = ns[1:]

        for ns in self.var('namespaces', default=[]):
            self.namespaces[ns.name] = [ns.uri, ns.get('schemaLocation', '')]

    def namespace(self, name):
        p = self.namespaces.get(name)
        return p or ['', '']
