_uid = 0


def uid():
    global _uid
    _uid += 1
    return _uid


class _Unit:
    def __init__(self):
        self.name = ''
        self.doc = ''
        self.type = ''

    def get(self, k, default=None):
        return vars(self).get(k, default)

    def dump(self):
        print(self.__class__.__name__ + ' (' + str(self.get('uid', '')) + ')')
        for k, v in sorted(vars(self).items()):
            print('\t', k, '=', repr(v))


class Unit(_Unit):
    """Parsing unit (AST node)."""

    def __init__(self, **kwargs):
        super().__init__()

        self.uid = uid()
        self.kind = ''
        self.args = []
        self.bases = []
        self.command = ''
        self.default = None
        self.module = ''
        self.optional = False
        self.parent = None
        self.supers = []
        self.types = []
        self.values = []

        for k, v in kwargs.items():
            setattr(self, k, v)


class Spec(_Unit):
    pass


class TypeSpec(Spec):
    """Type specification."""

    def __init__(self, **kwargs):
        super().__init__()

        self.bases = []
        self.values = []
        self.target = ''

        for k, v in kwargs.items():
            setattr(self, k, v)


class ObjectSpec(Spec):
    """Object specification."""

    def __init__(self, **kwargs):
        super().__init__()

        self.props = []
        self.extends = []

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.type = 'object'


class PropertySpec(Spec):
    """Property specification."""

    def __init__(self, **kwargs):
        super().__init__()

        self.default = None
        self.optional = False

        for k, v in kwargs.items():
            setattr(self, k, v)


class MethodSpec(Spec):
    """Method specification."""

    def __init__(self, **kwargs):
        super().__init__()

        self.action = ''
        self.arg = ''
        self.category = ''
        self.cmd = ''
        self.module = ''
        self.ret = ''

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.type = 'method'


class CliFunctionSpec(Spec):
    """CLI function specification."""

    def __init__(self, **kwargs):
        super().__init__()

        self.args = []
        self.command = ''
        self.module = ''
        self.subcommand = ''

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.type = 'clifunc'


class Stub:
    """Class stub."""

    def __init__(self, name, class_name):
        self.name = name
        self.class_name = class_name
        self.bases = []
        self.checked = False
        # a stub member is
        #   [p, None, type str or annotation node, source_line] = property
        #   [m, args annotation node, return annotation node, source_line] = method
        self.members = {}

    def dump(self):
        print('STUB: %s %s <= %s' % (self.name, self.bases, self.class_name))
        for k, v in self.members.items():
            print('\t', k, '=', repr(v))
