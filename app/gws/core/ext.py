"""Dummy decorators to support extension typing."""""


# @formatter:off
# flake8: noqa

class _ext:
    extName = ''

    def __init__(self, name):
        self.extName = self.extName + '.' + name
        print('init_ext', self.extName)

    def __call__(self, cls):
        print('call ext', cls)
        setattr(cls, 'extName', self.extName)
        return cls


class object:
    extName = 'gws.ext.object'
    class action(_ext): extName = 'gws.ext.object.action'
    class authMethod(_ext): extName = 'gws.ext.object.authMethod'
    class authProvider(_ext): extName = 'gws.ext.object.authProvider'
    class db(_ext): extName = 'gws.ext.object.db'
    class helper(_ext): extName = 'gws.ext.object.helper'
    class cli(_ext): extName = 'gws.ext.object.cli'
    class finder(_ext): extName = 'gws.ext.object.finder'
    class layer(_ext): extName = 'gws.ext.object.layer'
    class owsService(_ext): extName = 'gws.ext.object.owsService'
    class owsProvider(_ext): extName = 'gws.ext.object.owsProvider'
    class template(_ext): extName = 'gws.ext.object.template'


class config:
    class action(_ext): extName = 'gws.ext.config.action'
    class authMethod(_ext): extName = 'gws.ext.config.authMethod'
    class authProvider(_ext): extName = 'gws.ext.config.authProvider'
    class db(_ext): extName = 'gws.ext.config.db'
    class helper(_ext): extName = 'gws.ext.config.helper'
    class cli(_ext): extName = 'gws.ext.config.cli'
    class finder(_ext): extName = 'gws.ext.config.finder'
    class layer(_ext): extName = 'gws.ext.config.layer'
    class owsService(_ext): extName = 'gws.ext.config.owsService'
    class owsProvider(_ext): extName = 'gws.ext.config.owsProvider'
    class template(_ext): extName = 'gws.ext.config.template'


class props:
    class action(_ext): extName = 'gws.ext.props.action'
    class authMethod(_ext): extName = 'gws.ext.props.authMethod'
    class authProvider(_ext): extName = 'gws.ext.props.authProvider'
    class db(_ext): extName = 'gws.ext.props.db'
    class helper(_ext): extName = 'gws.ext.props.helper'
    class cli(_ext): extName = 'gws.ext.props.cli'
    class finder(_ext): extName = 'gws.ext.props.finder'
    class layer(_ext): extName = 'gws.ext.props.layer'
    class owsService(_ext): extName = 'gws.ext.props.owsService'
    class owsProvider(_ext): extName = 'gws.ext.props.owsProvider'
    class template(_ext): extName = 'gws.ext.props.template'


class command:
    class api(_ext): extName = 'gws.ext.command.api'
    class get(_ext): extName = 'gws.ext.command.get'
    class post(_ext): extName = 'gws.ext.command.post'
    class cli(_ext): extName = 'gws.ext.command.cli'

