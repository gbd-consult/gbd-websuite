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


def is_tag(cls):
    return issubclass(cls, _tag)


def is_name(s):
    return s.startswith('gws.ext.')


def name(cls):
    return cls.extName if issubclass(cls, _tag) else ''


class command:
    class api(_tag): extName = 'gws.ext.command.api'
    class get(_tag): extName = 'gws.ext.command.get'
    class post(_tag): extName = 'gws.ext.command.post'
    class cli(_tag): extName = 'gws.ext.command.cli'

##

class object:
    extName = 'gws.ext.object'
    class action(_tag): extName = 'gws.ext.object.action'
    class application(_tag): extName = 'gws.ext.object.application'
    class authMethod(_tag): extName = 'gws.ext.object.authMethod'
    class authMfa(_tag): extName = 'gws.ext.object.authMfa'
    class authProvider(_tag): extName = 'gws.ext.object.authProvider'
    class cli(_tag): extName = 'gws.ext.object.cli'
    class db(_tag): extName = 'gws.ext.object.db'
    class finder(_tag): extName = 'gws.ext.object.finder'
    class helper(_tag): extName = 'gws.ext.object.helper'
    class layer(_tag): extName = 'gws.ext.object.layer'
    class legend(_tag): extName = 'gws.ext.object.legend'
    class map(_tag): extName = 'gws.ext.object.map'
    class model(_tag): extName = 'gws.ext.object.model'
    class modelField(_tag): extName = 'gws.ext.object.modelField'
    class modelValidator(_tag): extName = 'gws.ext.object.modelValidator'
    class modelWidget(_tag): extName = 'gws.ext.object.modelWidget'
    class owsProvider(_tag): extName = 'gws.ext.object.owsProvider'
    class owsService(_tag): extName = 'gws.ext.object.owsService'
    class project(_tag): extName = 'gws.ext.object.project'
    class template(_tag): extName = 'gws.ext.object.template'


class config:
    extName = 'gws.ext.config'
    class action(_tag): extName = 'gws.ext.config.action'
    class application(_tag): extName = 'gws.ext.config.application'
    class authMethod(_tag): extName = 'gws.ext.config.authMethod'
    class authMfa(_tag): extName = 'gws.ext.config.authMfa'
    class authProvider(_tag): extName = 'gws.ext.config.authProvider'
    class cli(_tag): extName = 'gws.ext.config.cli'
    class db(_tag): extName = 'gws.ext.config.db'
    class finder(_tag): extName = 'gws.ext.config.finder'
    class helper(_tag): extName = 'gws.ext.config.helper'
    class layer(_tag): extName = 'gws.ext.config.layer'
    class legend(_tag): extName = 'gws.ext.config.legend'
    class map(_tag): extName = 'gws.ext.config.map'
    class model(_tag): extName = 'gws.ext.config.model'
    class modelField(_tag): extName = 'gws.ext.config.modelField'
    class modelValidator(_tag): extName = 'gws.ext.config.modelValidator'
    class modelWidget(_tag): extName = 'gws.ext.config.modelWidget'
    class owsProvider(_tag): extName = 'gws.ext.config.owsProvider'
    class owsService(_tag): extName = 'gws.ext.config.owsService'
    class project(_tag): extName = 'gws.ext.config.project'
    class template(_tag): extName = 'gws.ext.config.template'


class props:
    extName = 'gws.ext.props'
    class action(_tag): extName = 'gws.ext.props.action'
    class application(_tag): extName = 'gws.ext.props.application'
    class authMethod(_tag): extName = 'gws.ext.props.authMethod'
    class authMfa(_tag): extName = 'gws.ext.props.authMfa'
    class authProvider(_tag): extName = 'gws.ext.props.authProvider'
    class cli(_tag): extName = 'gws.ext.props.cli'
    class db(_tag): extName = 'gws.ext.props.db'
    class finder(_tag): extName = 'gws.ext.props.finder'
    class helper(_tag): extName = 'gws.ext.props.helper'
    class layer(_tag): extName = 'gws.ext.props.layer'
    class legend(_tag): extName = 'gws.ext.props.legend'
    class map(_tag): extName = 'gws.ext.props.map'
    class model(_tag): extName = 'gws.ext.props.model'
    class modelField(_tag): extName = 'gws.ext.props.modelField'
    class modelValidator(_tag): extName = 'gws.ext.props.modelValidator'
    class modelWidget(_tag): extName = 'gws.ext.props.modelWidget'
    class owsProvider(_tag): extName = 'gws.ext.props.owsProvider'
    class owsService(_tag): extName = 'gws.ext.props.owsService'
    class project(_tag): extName = 'gws.ext.props.project'
    class template(_tag): extName = 'gws.ext.props.template'
