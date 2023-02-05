import gws
import gws.spec
from . import error, log, types, util

import gws.types as t


def class_name(obj):
    return obj.__class__.__module__ + '.' + obj.__class__.__name__


##


class Object(types.IObject):
    pass


class Node(Object, types.INode):
    def initialize(self, config):
        self.config = config

        self.permissions = {}

        p = util.parse_acl(self.var('access'))
        if p:
            self.permissions[types.Access.use] = p
        p = self.var('permissions')
        if p:
            for k, v in vars(p).items():
                self.permissions[k] = util.parse_acl(v)
        p = self.permissions.get(gws.Access.use)
        if p:
            self.permissions.setdefault(gws.Access.read, p)

        # # since `super().configure` is mandatory in `configure` methods,
        # # let's automate this by collecting all super 'configure' methods upto 'Node'
        #
        mro = []
        for cls in type(self).mro():
            if cls == Node:
                break
            try:
                if 'configure' in vars(cls):
                    mro.append(cls)
            except TypeError:
                pass

        self.pre_configure()
        for cls in reversed(mro):
            cls.configure(self)  # type: ignore

    def var(self, key: str, default=None):
        val = util.get(self.config, key)
        return val if val is not None else default

    def create_child(self, classref, config=None):
        return self.root.create(classref, parent=self, config=config)

    def create_child_if_configured(self, classref, config=None):
        if not config:
            return None
        return self.root.create(classref, parent=self, config=config)

    def create_children(self, classref, configs):
        if not configs:
            return []
        return gws.compact(self.create_child(classref, cfg) for cfg in configs)

    def find_first(self, classref):
        return _find_first_in(self.root, self.children, classref)

    def find_all(self, classref):
        return _find_all_in(self.root, self.children, classref)


class Root(types.IRoot):
    def __init__(self):
        self.app = t.cast(types.IApplication, None)
        self.configErrors = []
        self._configStack = []
        self._cachedDescriptors: t.Dict[str, types.ExtObjectDescriptor] = {}
        self._objects: t.List['Node'] = []
        self._uidMap: t.Dict[str, 'Node'] = {}
        self._uidCount = 0

    def activate(self):
        for obj in self._objects:
            obj.activate()

    def initialize(self, obj, config):
        self._configStack.append(obj)

        try:
            t.cast(Node, obj).initialize(config)
            ok = True
        except Exception as exc:
            log.exception()
            self._config_error(exc)
            ok = False

        self._configStack.pop()
        return ok

    def post_initialize(self):
        for obj in reversed(self._objects):
            self._configStack = [obj]
            try:
                obj.post_configure()
            except Exception as exc:
                log.exception()
                self._config_error(exc)

    def get(self, uid, classref=None):
        obj = self._uidMap.get(uid)
        if obj and (not classref or _is_a(self, obj, classref)):
            return obj

    def find_first(self, classref):
        return _find_first_in(self, self._objects, classref)

    def find_all(self, classref):
        return _find_all_in(self, self._objects, classref)

    def create_application(self, config=None):

        obj = self.create('gws.base.application.Object')
        obj.uid = '0'
        obj.parent = self
        obj.children = []

        self._objects.append(obj)
        self._uidMap[obj.uid] = obj
        self.app = obj

        self.initialize(obj, util.to_data(config))

        return obj

    def create_shared(self, classref, config=None):
        if not config:
            cls = self.specs.get_class(classref)
            if not cls:
                raise error.Error(f'class {classref!r} not found')
            config = dict(uid=f'{cls.__module__}.{cls.__name__}.SHARED.VOID')

        config = util.to_data(config)

        uid = config.get('uid')
        if uid and uid in self._uidMap:
            return self._uidMap[uid]

        return self.create(classref, config=config)

    def create(self, classref, parent=None, config=None):
        config = util.to_data(config)

        obj = self._create(classref, config.get('type'))
        obj.uid = self._get_uid(config)
        obj.parent = parent
        obj.children = []

        log.debug(f'configuring {_object_name(obj)} in {_object_name(parent)}')
        ok = self.initialize(obj, config)
        if not ok:
            log.debug(f'FAILED {_object_name(obj)}')
            return

        self._objects.append(obj)
        self._uidMap[obj.uid] = obj

        if parent:
            parent.children.append(obj)

        return obj

    def _create(self, classref, typ=None):
        cls = self.specs.get_class(classref, typ)
        if not cls:
            raise error.Error(f'class {classref}:{typ} not found')

        obj = cls()
        obj.root = self
        obj.extName = getattr(cls, 'extName', '')
        obj.extType = getattr(cls, 'extType', '')

        return obj

    def _get_uid(self, config):
        if config.get('uid'):
            return config.get('uid')
        self._uidCount += 1
        return str(self._uidCount)

    def _config_error(self, exc):
        lines = [repr(exc)]

        for val in reversed(self._configStack):
            line = class_name(val)
            for p in 'title', 'type', 'uid':
                s = getattr(val, p, None)
                if s:
                    line += f' ({p}={s!r})'
                    break
            lines.append('in ' + line)

        self.configErrors.append(lines)


##

def _find_first_in(root, where, classref):
    ls = _find_all_in(root, where, classref)
    return ls[0] if ls else None


def _find_all_in(root, where, classref):
    cls, name, ext_name = root.specs.parse_classref(classref)
    if cls:
        return [obj for obj in where if isinstance(obj, cls)]
    if name:
        return [obj for obj in where if class_name(obj) == name]
    if ext_name:
        return [obj for obj in where if obj.extName.startswith(ext_name)]


def _is_a(root, obj, classref):
    cls, name, ext_name = root.specs.parse_classref(classref)
    if cls:
        return isinstance(obj, cls)
    if name:
        return class_name(obj) == name
    if ext_name:
        return obj.extName.startswith(ext_name)
    return False


##


def create_root_object(specs: types.ISpecRuntime) -> types.IRoot:
    r = Root()
    r.specs = specs
    return r


##


def _object_name(obj):
    if not obj:
        return '?'
    name = util.get(obj, 'extName', '') or class_name(obj)
    uid = util.get(obj, 'uid', '')
    if uid:
        name += '(' + str(uid) + ')'
    return name


##


def props(obj: object, user: types.IUser, *context) -> t.Optional[types.Props]:
    if not user.can_use(obj, *context):
        return None
    p = _make_props(obj, user)
    if p is None or util.is_data_object(p):
        return p
    if isinstance(p, dict):
        return types.Props(p)
    raise error.Error('invalid props type')


def _make_props(obj: t.Any, user: types.IUser):
    if util.is_atom(obj):
        return obj

    if user.acl_bit(obj, gws.Access.use) == gws.DENY:
        return None

    if isinstance(obj, Object):
        obj = obj.props(user)

    if util.is_data_object(obj):
        obj = vars(obj)

    if util.is_dict(obj):
        return util.compact({k: _make_props(v, user) for k, v in obj.items()})

    if util.is_list(obj):
        return util.compact([_make_props(v, user) for v in obj])

    return None
