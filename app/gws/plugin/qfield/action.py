import gws.base.action

from . import core

gws.ext.new.action('qfield')


class Config(gws.ConfigWithAccess):
    """QField action."""

    packages: list[core.PackageConfig]


class Props(gws.base.action.Props):
    pass


class Object(gws.Node):
    packages: dict[str, core.Package]

    def configure(self):
        self.packages = {
            p.name: p
            for p in self.create_children(core.Package, self.cfg('packages'))
        }


