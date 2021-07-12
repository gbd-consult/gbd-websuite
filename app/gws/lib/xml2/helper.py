import gws
import gws.types as t

from . import namespaces


class NamespaceConfig(gws.Data):
    """XML namespace configuration"""

    name: str  #: namespace name
    uri: gws.Url  #: namespace uri
    schemaLocation: t.Optional[gws.Url]  #: namespace schema location


@gws.ext.Config('helper.xml')
class Config(gws.Config):
    """XML settings"""

    namespaces: t.Optional[t.List[NamespaceConfig]]  #: custom namespaces


@gws.ext.Object('helper.xml')
class Object(gws.Node):
    def configure(self):

        self.namespaces = {}

        for ns in namespaces.ALL:
            self.namespaces[ns[0]] = ns[1:]

        for ns in self.var('namespaces', default=[]):
            self.namespaces[ns.name] = [ns.uri, ns.get('schemaLocation', '')]

    def namespace(self, name):
        p = self.namespaces.get(name)
        return p or ['', '']
