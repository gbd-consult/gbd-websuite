"""Dummy decorators to support extension typing."""""


# @formatter:off
# flake8: noqa

class _tag:
    extName = ''

    def __init__(self, typ):
        self.extName = self.extName + '.' + typ
        self.extType = typ

    def __call__(self, cls):
        setattr(cls, 'extName', self.extName)
        setattr(cls, 'extType', self.extType)
        return cls


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


def name(obj):
    if isinstance(obj, str) and obj.startswith('gws.ext.'):
        return name
    if isinstance(obj, type) and issubclass(obj, _classTag):
        return obj.extName

##


TYPES = [
    "action",
    "application",
    "authMethod",
    "authMfa",
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
    "project",
    "template",
]


class command:
    class api(_methodTag): pass
    class cli(_methodTag): pass
    class get(_methodTag): pass
    class post(_methodTag): pass


class new:
    def action(*a): pass
    def application(*a): pass
    def authMethod(*a): pass
    def authMfa(*a): pass
    def authProvider(*a): pass
    def authSessionManager(*a): pass
    def cli(*a): pass
    def databaseProvider(*a): pass
    def finder(*a): pass
    def helper(*a): pass
    def layer(*a): pass
    def legend(*a): pass
    def map(*a): pass
    def model(*a): pass
    def modelField(*a): pass
    def modelValidator(*a): pass
    def modelValue(*a): pass
    def modelWidget(*a): pass
    def owsProvider(*a): pass
    def owsService(*a): pass
    def project(*a): pass
    def template(*a): pass


class object:
    class action (_classTag): extName = 'gws.ext.object.action'
    class application (_classTag): extName = 'gws.ext.object.application'
    class authMethod (_classTag): extName = 'gws.ext.object.authMethod'
    class authMfa (_classTag): extName = 'gws.ext.object.authMfa'
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
    class project (_classTag): extName = 'gws.ext.object.project'
    class template (_classTag): extName = 'gws.ext.object.template'


class config:
    class action (_classTag): extName = 'gws.ext.config.action'
    class application (_classTag): extName = 'gws.ext.config.application'
    class authMethod (_classTag): extName = 'gws.ext.config.authMethod'
    class authMfa (_classTag): extName = 'gws.ext.config.authMfa'
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
    class project (_classTag): extName = 'gws.ext.config.project'
    class template (_classTag): extName = 'gws.ext.config.template'


class props:
    class action (_classTag): extName = 'gws.ext.props.action'
    class application (_classTag): extName = 'gws.ext.props.application'
    class authMethod (_classTag): extName = 'gws.ext.props.authMethod'
    class authMfa (_classTag): extName = 'gws.ext.props.authMfa'
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
    class project (_classTag): extName = 'gws.ext.props.project'
    class template (_classTag): extName = 'gws.ext.props.template'
