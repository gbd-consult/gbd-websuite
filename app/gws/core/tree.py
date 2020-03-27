import importlib

from . import util, error, log, debug
import gws.types as t

_UIDS = set()


#:export IObject
class Object(t.IObject):
    access: t.Access
    config: t.Config
    parent: t.IObject
    root: t.IRootObject

    def __init__(self):
        self.children: t.List[t.IObject] = []
        self.klass = _class_name(self.__class__)
        self.uid: str = ''
        for a in 'access', 'config', 'parent', 'root':
            setattr(self, a, None)

    @property
    def props(self) -> t.Props:
        return t.cast(t.Props, None)

    def is_a(self, klass):
        if isinstance(klass, type):
            return isinstance(self, klass)
        return self.klass == klass or self.klass.startswith(klass + '.')

    def _new_uid(self, uid):
        global _UIDS

        n = 0
        u = uid
        while u in _UIDS:
            n += 1
            u = uid + str(n)
        _UIDS.add(u)
        return u

    def _auto_uid(self):
        u = self.var('uid')
        if u:
            return u
        u = self.var('title')
        if u:
            return util.as_uid(u)
        return self.klass.replace('.', '_')

    def set_uid(self, uid):
        global _UIDS

        if not uid or uid == self.uid:
            return

        with util.global_lock():
            if self.uid:
                _UIDS.discard(self.uid)
            self.uid = self._new_uid(uid)

    def initialize(self, cfg):
        self.config = cfg
        self.access = self.var('access')
        self.set_uid(self._auto_uid())

        log.debug(f'BEGIN configure: {self.klass}')

        try:
            self.configure()
        except Exception as e:
            # try to provide a clue where this happened
            msg = '%s\nin %s' % (_exc_name_for_error(e), _object_name_for_error(self))
            raise error.Error(msg)

        log.debug(f'END configure: {self.klass} uid={self.uid}')

    def configure(self):
        # this is intended to be overridden
        pass

    def post_initialize(self):
        try:
            self.post_configure()
        except Exception as e:
            msg = '%s\nin %s' % (_exc_name_for_error(e), _object_name_for_error(self))
            raise error.Error(msg)

        for c in self.children:
            c.post_initialize()

    def post_configure(self):
        # this is intended to be overridden
        pass

    def var(self, key, default=None, parent=False):
        v = util.get(self.config, key)
        if v is not None:
            return v
        if parent and self.parent:
            return self.parent.var(key, default, parent=True)
        return default

    def create_child(self, klass, cfg) -> t.IObject:
        return self.append_child(self.root.create_object(klass, cfg, parent=self))

    def append_child(self, obj: t.IObject) -> t.IObject:
        obj.parent = self
        self.children.append(obj)
        return obj

    def get_children(self, klass) -> t.List[t.IObject]:
        return list(_find_all(self.children, klass))

    def get_closest(self, klass) -> t.IObject:
        if self.parent:
            if self.parent.is_a(klass):
                return self.parent
            return self.parent.get_closest(klass)

    def props_for(self, user) -> t.Optional[dict]:
        if not user.can_use(self):
            return None
        return _make_props(self.props, user)


#:export IRootObject
class RootObject(Object, t.IRootObject):
    application: t.IApplication
    validator: t.SpecValidator

    def __init__(self):
        super().__init__()
        self.all_types = {}
        self.all_objects = []
        self.shared_objects = {}
        self.root = self
        for a in 'application', 'validator':
            setattr(self, a, None)

    def create(self, klass, cfg=None):
        cfg = _to_config(cfg)
        oo = self._create(klass, cfg)
        oo.root = self
        return oo

    def create_object(self, klass, cfg, parent=None):
        cfg = _to_config(cfg)
        obj = self.create(klass, cfg)
        obj.parent = parent
        obj.initialize(cfg)
        self.all_objects.append(obj)
        return obj

    def create_unbound_object(self, klass, cfg):
        cfg = _to_config(cfg)
        obj = self.create(klass, cfg)
        obj.initialize(cfg)
        return obj

    def create_shared_object(self, klass, uid, cfg):
        cfg = _to_config(cfg)
        uid = _class_name(klass).replace('.', '_') + '_' + util.as_uid(uid)

        if uid in self.shared_objects:
            # log.debug(f'SHARED: FOUND {klass} {uid}')
            return self.shared_objects[uid]

        with util.global_lock():
            log.debug(f'SHARED: create {klass} {uid}')
            obj = self.create_object(klass, util.merge(cfg, {'uid': uid}))
            self.shared_objects[uid] = obj

        return obj

    def find_all(self, klass=None) -> t.List[t.IObject]:
        return list(_find_all(self.all_objects, klass))

    def find_first(self, klass) -> t.IObject:
        for p in _find_all(self.all_objects, klass):
            return p

    def find(self, klass, uid) -> t.IObject:
        return _find(self.all_objects, klass, uid)

    def _create(self, klass, cfg):
        if isinstance(klass, type):
            return klass()
        if cfg.type:
            klass += '.' + cfg.type
        if klass not in self.all_types:
            self.all_types[klass] = _load_class(klass)
        return self.all_types[klass]()


def _to_config(cfg):
    if util.is_data_object(cfg):
        return cfg
    if isinstance(cfg, dict):
        return t.Data(cfg)
    if not cfg:
        return t.Data()
    return cfg


def _load_class(klass):
    try:
        mod = importlib.import_module(klass)
    except Exception as e:
        raise error.Error(f'import of {klass!r} failed') from e
    try:
        return mod.Object
    except Exception as e:
        raise error.Error(f'object not found in {klass!r}') from e


def _find(nodes, klass, uid):
    if not uid:
        return
    for obj in nodes:
        if obj.uid == uid and obj.is_a(klass):
            return obj


def _find_all(nodes, klass):
    if not klass:
        yield from nodes
    else:
        for obj in nodes:
            if obj.is_a(klass):
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


def _make_props(obj, user):
    if obj is None or isinstance(obj, (int, float, bool, str, bytes)):
        return obj

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

    if util.has(obj, 'props_for'):
        return _make_props(obj.props_for(user), user)

    if util.has(obj, 'props'):
        return _make_props(obj.props, user)

    raise ValueError(f'make_props failed for {obj.__class__}')
