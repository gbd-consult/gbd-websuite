import gws.core.tree
import gws.types.spec
import gws.common.application
import gws.server.monitor

from . import parser, error, spec


class Object(gws.core.tree.RootObject):
    def __init__(self):
        super().__init__()
        self.application: gws.common.application.Object = None
        self.monitor: gws.server.monitor.Object = None
        self.action_commands = []
        self.action_validator = None
        self.app_data = {}

    def configure(self):
        super().configure()

        self.monitor = self.add_child(gws.server.monitor.Object, {})
        self.application = self.add_child(gws.common.application.Object, self.config)
        self.action_commands = spec.action_commands()
        self.action_validator = spec.action_validator()

    def validate_action(self, category, cmd, arg):
        if cmd not in self.action_commands:
            raise error.DispatchError('not found', cmd)

        cc = self.action_commands[cmd]
        if cc['category'] != category:
            raise error.DispatchError('wrong command category', category)

        if category == 'api':
            try:
                arg = self.action_validator.get(arg, cc['arg'])
            except gws.types.spec.Error as e:
                gws.log.exception()
                raise error.DispatchError('invalid parameters') from e

        return cc['action'], cc['method'], arg
