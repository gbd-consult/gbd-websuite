import gws
import gws.spec
from . import error, log, types, util

import gws.types as t


def class_name(obj):
    return obj.__class__.__module__ + '.' + obj.__class__.__name__


##


class Object(types.IObject):
    def __repr__(self):
        r = getattr(self, 'extName', None) or class_name(self)
        s = getattr(self, 'title', None)
        if s:
            r += f' title={s!r}'
        s = getattr(self, 'uid', None)
        if s:
            r += f' uid={s}'
        return '<' + r + ' ' + hex(id(self)) + '>'


class Node(Object, types.INode):
    def initialize(self, config):
        self.config = config
        self.permissions = self._confure_permissions()
        _super_invoke(self, 'pre_configure')
        _super_invoke(self, 'configure')

    def _confure_permissions(self):
        perms = {}

        p = self.cfg('access')
        if p:
            perms[types.Access.use] = util.parse_acl(p)

        p = self.cfg('permissions')
        if p:
            for k, v in vars(p).items():
                a = util.parse_acl(v)
                if k == 'edit':
                    perms[gws.Access.write] = perms[gws.Access.create] = perms[gws.Access.delete] = a
                else:
                    perms[t.cast(gws.Access, k)] = a

        p = perms.get(gws.Access.use)
        if p:
            perms.setdefault(gws.Access.read, p)

        return perms

    def cfg(self, key: str, default=None):
        val = util.get(self.config, key)
        return val if val is not None else default

    def create_child(self, classref, config=None, **kwargs):
        return self.root.create(classref, parent=self, config=config, **kwargs)

    def create_child_if_configured(self, classref, config=None, **kwargs):
        if not config:
            return None
        return self.root.create(classref, parent=self, config=config, **kwargs)

    def create_children(self, classref, configs, **kwargs):
        if not configs:
            return []
        return gws.compact(self.create_child(classref, cfg, **kwargs) for cfg in configs)

    def find_first(self, classref):
        return _find_first_in(self.root, self.children, classref)

    def find_all(self, classref):
        return _find_all_in(self.root, self.children, classref)


class Root(types.IRoot):
    def __init__(self):
        self.app = t.cast(types.IApplication, None)
        self.configErrors = []
        self._configStack = []
        self._cachedDescriptors: dict[str, types.ExtObjectDescriptor] = {}
        self._objects: list['Node'] = []
        self._uidMap: dict[str, 'Node'] = {}
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
                _super_invoke(obj, 'post_configure')
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

    def create_application(self, config=None, **kwargs):
        config = _to_config(config, kwargs)

        obj = self._alloc('gws.base.application.Object')
        obj.uid = '0'
        obj.parent = self
        obj.children = []

        self._objects.append(obj)
        self._uidMap[obj.uid] = obj
        self.app = obj

        self.initialize(obj, config)

        return obj

    def create_shared(self, classref, config=None, **kwargs):
        config = _to_config(config, kwargs)

        uid = config.get('uid')
        if uid and uid in self._uidMap:
            return self._uidMap[uid]

        return self._create(classref, None, config)

    def create(self, classref, parent=None, config=None, **kwargs):
        config = _to_config(config, kwargs)
        return self._create(classref, parent, config)

    def create_temporary(self, classref, config=None, **kwargs):
        config = _to_config(config, kwargs)
        return self._create(classref, None, config, temp=True)

    def _create(self, classref, parent, config, temp=False):
        obj = self._alloc(classref, config.get('type'))
        obj.uid = self._get_uid(config)
        obj.parent = parent
        obj.children = []

        log.debug(f'configuring {obj!r} in ' + (repr(parent) if parent else '<root>'))
        ok = self.initialize(obj, config)
        if not ok:
            log.debug(f'FAILED {obj!r}')
            return

        if not temp:
            self._objects.append(obj)
            self._uidMap[obj.uid] = obj

        if parent:
            parent.children.append(obj)

        return obj

    def _alloc(self, classref, typ=None):
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
            lines.append('in ' + repr(val))

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


def _super_invoke(obj, method):
    # since `super().configure` is mandatory in `configure` methods,
    # let's automate this by collecting all super 'configure' methods

    mro = []

    for cls in type(obj).mro():
        try:
            if method in vars(cls):
                mro.append(cls)
        except TypeError:
            pass

    for cls in reversed(mro):
        getattr(cls, method)(obj)


def _to_config(config, defaults):
    return gws.merge(gws.Data(), defaults, config)


##


def create_root_object(specs: types.ISpecRuntime) -> types.IRoot:
    r = Root()
    r.specs = specs
    return r


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
