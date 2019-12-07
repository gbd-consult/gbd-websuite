import gws.core.tree
import gws.types.spec
import gws.server.monitor

from . import error, spec


class Object(gws.core.tree.RootObject):
    def __init__(self):
        super().__init__()
        self.application = None
        self.monitor: gws.server.monitor.Object = None
        self.validator: gws.types.spec.Validator = None
        self.app_data = {}

    def configure(self):
        super().configure()

        self.monitor = self.add_child(gws.server.monitor.Object, {})
        self.application = self.add_child('gws.common.application', self.config)
        self.validator = spec.validator()

    def validate_action(self, category, cmd, payload):
        cc = self.validator.method_spec(cmd)
        if not cc:
            raise error.DispatchError('not found', cmd)

        cat = cc['category']
        if cat == 'http' and category.startswith('http'):
            cat = category
        if category != cat:
            raise error.DispatchError('wrong command category', category)

        if cc['arg']:
            try:
                payload = self.validator.read_value(payload, cc['arg'], strict=(cat == 'api'))
            except gws.types.spec.Error as e:
                gws.log.exception()
                raise error.DispatchError('invalid parameters') from e

        return cc['action'], cc['name'], payload
