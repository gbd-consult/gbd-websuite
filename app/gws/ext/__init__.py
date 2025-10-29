"""Dummy decorators to support extension typing."""


class _classTag:
    extName = ''

    def __init__(self, typ):
        pass

    def __call__(self, target):
        return target


class _methodTag:
    def __init__(self, typ):
        pass

    def __call__(self, target):
        return target


def name_for(obj: str | type) -> str | None:
    if isinstance(obj, str) and obj.startswith('gws.ext.'):
        return obj
    if isinstance(obj, type) and issubclass(obj, _classTag):
        return obj.extName


# fmt: off

##


TYPES = [
    "action",
    "application",
    "authMethod",
    "authMultiFactorAdapter",
    "authProvider",
    "authSessionManager",
    "cli",
    "databaseProvider",
    "finder",
    "helper",
    "layer",
    "legend",
    "map",
    "model",
    "modelField",
    "modelValidator",
    "modelValue",
    "modelWidget",
    "owsProvider",
    "owsService",
    "printer",
    "project",
    "storageProvider",
    "template",
]


class command:
    class api(_methodTag): pass
    class cli(_methodTag): pass
    class get(_methodTag): pass
    class post(_methodTag): pass
    class raw(_methodTag): pass


class _new:
    def action(self, *args): pass
    def application(self, *args): pass
    def authMethod(self, *args): pass
    def authMultiFactorAdapter(self, *args): pass
    def authProvider(self, *args): pass
    def authSessionManager(self, *args): pass
    def cli(self, *args): pass
    def databaseProvider(self, *args): pass
    def finder(self, *args): pass
    def helper(self, *args): pass
    def layer(self, *args): pass
    def legend(self, *args): pass
    def map(self, *args): pass
    def model(self, *args): pass
    def modelField(self, *args): pass
    def modelValidator(self, *args): pass
    def modelValue(self, *args): pass
    def modelWidget(self, *args): pass
    def owsProvider(self, *args): pass
    def owsService(self, *args): pass
    def printer(self, *args): pass
    def project(self, *args): pass
    def storageProvider(self, *args): pass
    def template(self, *args): pass


new = _new()


class object:
    class action (_classTag): extName = 'gws.ext.object.action'
    class application (_classTag): extName = 'gws.ext.object.application'
    class authMethod (_classTag): extName = 'gws.ext.object.authMethod'
    class authMultiFactorAdapter (_classTag): extName = 'gws.ext.object.authMultiFactorAdapter'
    class authProvider (_classTag): extName = 'gws.ext.object.authProvider'
    class authSessionManager (_classTag): extName = 'gws.ext.object.authSessionManager'
    class cli (_classTag): extName = 'gws.ext.object.cli'
    class databaseProvider (_classTag): extName = 'gws.ext.object.databaseProvider'
    class finder (_classTag): extName = 'gws.ext.object.finder'
    class helper (_classTag): extName = 'gws.ext.object.helper'
    class layer (_classTag): extName = 'gws.ext.object.layer'
    class legend (_classTag): extName = 'gws.ext.object.legend'
    class map (_classTag): extName = 'gws.ext.object.map'
    class model (_classTag): extName = 'gws.ext.object.model'
    class modelField (_classTag): extName = 'gws.ext.object.modelField'
    class modelValidator (_classTag): extName = 'gws.ext.object.modelValidator'
    class modelValue (_classTag): extName = 'gws.ext.object.modelValue'
    class modelWidget (_classTag): extName = 'gws.ext.object.modelWidget'
    class owsProvider (_classTag): extName = 'gws.ext.object.owsProvider'
    class owsService (_classTag): extName = 'gws.ext.object.owsService'
    class printer (_classTag): extName = 'gws.ext.object.printer'
    class project (_classTag): extName = 'gws.ext.object.project'
    class storageProvider (_classTag): extName = 'gws.ext.object.storageProvider'
    class template (_classTag): extName = 'gws.ext.object.template'


class config:
    class action (_classTag): extName = 'gws.ext.config.action'
    class application (_classTag): extName = 'gws.ext.config.application'
    class authMethod (_classTag): extName = 'gws.ext.config.authMethod'
    class authMultiFactorAdapter (_classTag): extName = 'gws.ext.config.authMultiFactorAdapter'
    class authProvider (_classTag): extName = 'gws.ext.config.authProvider'
    class authSessionManager (_classTag): extName = 'gws.ext.config.authSessionManager'
    class cli (_classTag): extName = 'gws.ext.config.cli'
    class databaseProvider (_classTag): extName = 'gws.ext.config.databaseProvider'
    class finder (_classTag): extName = 'gws.ext.config.finder'
    class helper (_classTag): extName = 'gws.ext.config.helper'
    class layer (_classTag): extName = 'gws.ext.config.layer'
    class legend (_classTag): extName = 'gws.ext.config.legend'
    class map (_classTag): extName = 'gws.ext.config.map'
    class model (_classTag): extName = 'gws.ext.config.model'
    class modelField (_classTag): extName = 'gws.ext.config.modelField'
    class modelValidator (_classTag): extName = 'gws.ext.config.modelValidator'
    class modelValue (_classTag): extName = 'gws.ext.config.modelValue'
    class modelWidget (_classTag): extName = 'gws.ext.config.modelWidget'
    class owsProvider (_classTag): extName = 'gws.ext.config.owsProvider'
    class owsService (_classTag): extName = 'gws.ext.config.owsService'
    class printer (_classTag): extName = 'gws.ext.config.printer'
    class project (_classTag): extName = 'gws.ext.config.project'
    class storageProvider (_classTag): extName = 'gws.ext.config.storageProvider'
    class template (_classTag): extName = 'gws.ext.config.template'


class props:
    class action (_classTag): extName = 'gws.ext.props.action'
    class application (_classTag): extName = 'gws.ext.props.application'
    class authMethod (_classTag): extName = 'gws.ext.props.authMethod'
    class authMultiFactorAdapter (_classTag): extName = 'gws.ext.props.authMultiFactorAdapter'
    class authProvider (_classTag): extName = 'gws.ext.props.authProvider'
    class authSessionManager (_classTag): extName = 'gws.ext.props.authSessionManager'
    class cli (_classTag): extName = 'gws.ext.props.cli'
    class databaseProvider (_classTag): extName = 'gws.ext.props.databaseProvider'
    class finder (_classTag): extName = 'gws.ext.props.finder'
    class helper (_classTag): extName = 'gws.ext.props.helper'
    class layer (_classTag): extName = 'gws.ext.props.layer'
    class legend (_classTag): extName = 'gws.ext.props.legend'
    class map (_classTag): extName = 'gws.ext.props.map'
    class model (_classTag): extName = 'gws.ext.props.model'
    class modelField (_classTag): extName = 'gws.ext.props.modelField'
    class modelValidator (_classTag): extName = 'gws.ext.props.modelValidator'
    class modelValue (_classTag): extName = 'gws.ext.props.modelValue'
    class modelWidget (_classTag): extName = 'gws.ext.props.modelWidget'
    class owsProvider (_classTag): extName = 'gws.ext.props.owsProvider'
    class owsService (_classTag): extName = 'gws.ext.props.owsService'
    class printer (_classTag): extName = 'gws.ext.props.printer'
    class project (_classTag): extName = 'gws.ext.props.project'
    class storageProvider (_classTag): extName = 'gws.ext.props.storageProvider'
    class template (_classTag): extName = 'gws.ext.props.template'
