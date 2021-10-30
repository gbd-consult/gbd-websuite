import gws
import gws.types as t

from . import namespaces, types


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
    fallback_namespace: types.Namespace
    namespaces: t.Dict[str, types.Namespace]

    def configure(self):
        self.fallback_namespace = types.Namespace(
            name='gws',
            uri='http://gbd-websuite.de/namespaces/gws',
            schema='', )

        self.namespaces = {
            self.fallback_namespace.name: self.fallback_namespace,
        }

        for name, uri, schema in namespaces.ALL:
            self.namespaces[name] = types.Namespace(name=name, uri=uri, schema=schema)
        for ns in self.var('namespaces', default=[]):
            self.namespaces[ns.name] = types.Namespace(name=ns.name, uri=ns.uri, schema=ns.get('schemaLocation', ''))

    def namespace(self, name) -> t.Optional[types.Namespace]:
        return self.namespaces.get(name)
