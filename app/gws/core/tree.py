import sys

import gws.types as t
from . import error, log, types, util


class Object(types.IObject):
    access: t.Optional[types.Access]
    uid: str
    title: str
    class_name: str
    ext_type: str
    config_error: str

    @property
    def props(self) -> types.Props:
        return t.cast(types.Props, None)

    def __init__(self, klass: types.Klass = None, ext_type: str = None):
        self.access = None
        self.class_name = _class_name(klass or self.__class__)
        self.ext_type = ext_type or ''
        self.title = ''
        self.uid = ''
        self.config_error = ''

    def is_a(self, klass: types.Klass):
        if isinstance(klass, type):
            return isinstance(self, klass)
        return self.class_name == klass or self.class_name.startswith(klass + '.')

    def props_for(self, user: types.IUser) -> t.Optional[types.Props]:
        if not user.can_use(self):
            return None
        return types.Props(_make_props(self.props, user))

    def configure(self):
        # this is intended to be overridden
        pass

    def post_configure(self):
        # this is intended to be overridden
        pass


class Node(Object, types.INode):
    root: 'RootObject'
    parent: 'Node'
    config: types.Config

    def __init__(self, klass: types.Klass = None, ext_type: str = None):
        super().__init__(klass, ext_type)
        self.children: t.List['Node'] = []
        for s in ('root', 'parent', 'config'):
            setattr(self, s, None)

    def initialize(self, cfg):
        self.config = cfg
        self.access = self.var('access')
        self.set_uid(_auto_uid(self))

        log.debug(f'BEGIN config {self.class_name}')

        # call all super 'configure' methods
        mro = [cls for cls in type(self).mro() if hasattr(cls, 'configure')]
        try:
            for cls in reversed(mro):
                cls.configure(self)  # type: ignore
        except Exception as exc:
            self.config_error = _exc_name_for_error(exc)
            log.error(self.config_error)
            log.exception()

        log.debug(f'END config {self.class_name} uid={self.uid}')

    def set_uid(self, uid: str):
        _set_uid(self, uid)

    def var(self, key: str, default=None, with_parent=False):
        v = util.get(self.config, key)
        if v is not None:
            return v
        if with_parent and self.parent:
            return self.parent.var(key, default, with_parent=True)
        return default

    def create_child(self, klass: types.Klass, cfg: t.Optional[t.Any]) -> 'Node':
        return self.append_child(self.root.create_node(klass, cfg, parent=self))

    def append_child(self, obj: 'Node') -> 'Node':
        obj.parent = self
        self.children.append(obj)
        return obj

    def get_closest(self, klass: types.Klass) -> t.Optional['Node']:
        obj = self.parent
        while obj:
            if obj.is_a(klass):
                return obj
            obj = obj.parent
        return None


class RootObject(Object, types.IRootObject):
    application: types.IApplication
    specs: types.ISpecRuntime

    def __init__(self):
        super().__init__()
        self.all_types = {}
        self.all_nodes = []
        self.shared_objects = {}
        self.root = self

    def post_initialize(self):
        for obj in reversed(self.all_nodes):
            try:
                obj.post_configure()
            except Exception as exc:
                obj.config_error = _exc_name_for_error(exc)
                log.error(obj.config_error)
                log.exception()

    def create_application(self, klass, cfg):
        app = self._create(klass)
        app.root = self
        self.all_nodes.append(app)
        self.application = app
        t.cast('Node', self.application).initialize(cfg)

    def create_node(self, klass, cfg, parent=None):
        cfg = _to_config(cfg)
        obj = self._create(klass, cfg)
        obj.parent = parent
        obj.initialize(cfg)
        self.all_nodes.append(obj)
        return obj

    def create_object(self, klass, cfg):
        cfg = _to_config(cfg)
        obj = self._create(klass, cfg)
        obj.initialize(cfg)
        return obj

    def create_shared_object(self, klass, uid, cfg):
        cfg = _to_config(cfg)
        uid = util.as_uid(_class_name(klass) + '_' + uid)

        if uid in self.shared_objects:
            # log.debug(f'SHARED: FOUND {klass} {uid}')
            return self.shared_objects[uid]

        with util.global_lock():
            log.debug(f'create_shared_object: klass={klass!r} uid={uid!r}')
            obj = self.create_node(klass, util.merge(cfg, uid=uid))
            self.shared_objects[uid] = obj

        return obj

    def find_all(self, klass: types.Klass = None, uid: str = None, ext_type: str = None) -> t.List[types.INode]:
        return list(_find_all(self.all_nodes, klass, uid, ext_type))

    def find(self, klass: types.Klass = None, uid: str = None, ext_type: str = None) -> t.Optional[types.INode]:
        for p in _find_all(self.all_nodes, klass, uid, ext_type):
            return p
        return None

    def _create(self, klass, cfg=None):
        obj = self._create2(klass, _to_config(cfg))
        obj.root = self
        return obj

    def _create2(self, klass, cfg):
        if isinstance(klass, type):
            return klass()

        # gws.ext.action + type=auth -> gws.ext.action.auth.Object
        c = str(klass).split('.')
        ext_type = cfg.get('type')
        if ext_type:
            c.append(ext_type)
        if not c[-1][0].isupper():
            c.append('Object')
        class_name = '.'.join(c)

        if class_name in self.all_types:
            desc = self.all_types[class_name]
        else:
            desc = load_ext(self.specs, class_name)
            if not desc:
                raise _error(self, ValueError(f'class not found: {class_name!r}'))
            self.all_types[class_name] = desc

        cls = desc.class_ptr
        return cls(desc.name, desc.ext_type)


def load_ext(specs, class_name) -> types.ExtObjectDescriptor:
    desc = specs.ext_object_descriptor(class_name)
    if not desc:
        raise error.Error(f'load_ext: class {class_name!r} not found')
    if desc.module_name in sys.modules:
        log.debug(f'load_ext: {class_name!r}: found')
        mod = sys.modules[desc.module_name]
    else:
        log.debug(f'load_ext: {class_name!r}: loading from spec: {desc!r}')
        mod = util.import_from_path(desc.module_path, desc.module_name)
    desc.class_ptr = getattr(mod, desc.ident)
    return desc


def _to_config(cfg):
    if util.is_data_object(cfg):
        return cfg
    if isinstance(cfg, dict):
        return types.Config(cfg)
    if not cfg:
        return types.Config()
    return cfg


def _find_all(objects, klass, uid, ext_type):
    for obj in objects:
        ok = (
                (not klass or obj.is_a(klass))
                and (not uid or obj.uid == uid)
                and (not ext_type or obj.ext_type == ext_type)
        )
        if ok:
            yield obj


def _class_name(s):
    if isinstance(s, str):
        return s
    if not isinstance(s, type):
        s = s.__class__
    mod = s.__module__
    name = s.__name__
    if name == 'Object':
        return mod
    return mod + '.' + name


def _exc_name_for_error(e):
    cls = 'Error'
    try:
        cls = e.__class__.__name__
    except:
        pass
    a = '?'
    try:
        a = e.args[0]
    except:
        pass
    if isinstance(e, error.Error):
        return a
    return '%s: %s' % (cls, a)


def _object_name_for_error(x):
    cls = getattr(x, 'klass', '')
    if not cls:
        try:
            cls = x.__class__.__name__
        except:
            cls = 'object'

    uid = getattr(x, 'uid', '')
    if uid:
        return '%s(%s)' % (cls, uid)

    return cls


def _error(obj, exc):
    msg = '%s\nin %s' % (_exc_name_for_error(exc), _object_name_for_error(obj))
    raise error.Error(msg)


def _make_props(obj, user):
    if obj is None or isinstance(obj, (int, float, bool, str, bytes)):
        return obj

    if isinstance(obj, Object):
        return _make_props(obj.props_for(user), user)

    if util.is_data_object(obj):
        obj = vars(obj)

    if isinstance(obj, dict):
        ls = {}
        for k, v in obj.items():
            v = _make_props(v, user)
            if v is not None:
                ls[k] = v
        return ls

    if isinstance(obj, (list, tuple)):
        ls = []
        for v in obj:
            v = _make_props(v, user)
            if v is not None:
                ls.append(v)
        return ls

    if util.has(obj, 'props'):
        return _make_props(util.get(obj, 'props'), user)

    if obj:
        return str(obj)


##


_UIDS: t.Set[str] = set()


def _new_uid(uid):
    global _UIDS

    n = 0
    u = uid
    while u in _UIDS:
        n += 1
        u = uid + str(n)
    _UIDS.add(u)
    return u


def _auto_uid(obj):
    u = obj.var('uid')
    if u:
        return u
    u = obj.var('title')
    if u:
        return util.as_uid(u)
    return obj.class_name.replace('.', '_')


def _set_uid(obj, uid):
    global _UIDS

    if not uid or uid == obj.uid:
        return

    with util.global_lock():
        if obj.uid:
            _UIDS.discard(obj.uid)
        obj.uid = _new_uid(uid)
