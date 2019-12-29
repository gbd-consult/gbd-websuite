import importlib

from . import util, error, log
import gws.types as t

_uids = set()


def _new_uid(uid):
    n = 0
    u = uid
    while u in _uids:
        n += 1
        u = uid + str(n)
    _uids.add(u)
    return u

#:stub Object
class Object:
    def __init__(self):
        self.children = []
        self.config = None
        self.parent = None
        self.root: t.RootObject = None
        self.uid = ''

        self.access = None
        self.klass = _class_name(self.__class__)
        self.defaults = None

    def is_a(self, klass):
        if isinstance(klass, type):
            return isinstance(self, klass)
        if self.klass == klass:
            return True
        return self.klass.startswith(klass + '.')

    def set_uid(self, uid):
        with util.global_lock:
            if self.uid != uid:
                self.uid = _new_uid(uid)

    @property
    def auto_uid(self) -> str:
        u = self.var('uid')
        if u:
            return u
        u = self.var('title')
        if u:
            return util.as_uid(u)
        return self.klass.replace('.', '_')

    def initialize(self, cfg):
        self.config = cfg

        uid = self.auto_uid
        if uid:
            self.set_uid(uid)

        self.access = self.var('access')
        try:
            log.debug(f'configure: {self.klass} uid={self.uid}')
            self.configure()
        except Exception as e:
            # try to provide a clue where this happened
            msg = '%s\nin %s' % (_exc_name_for_error(e), _object_name_for_error(self))
            raise error.Error(msg)

    def configure(self):
        # this is intended to be overridden
        pass

    def var(self, key, default=None, parent=False):
        v = util.get(self.config, key)
        if v is not None:
            return v
        if parent and self.parent:
            return self.parent.var(key, default, parent=True)
        return default

    def create_object(self, klass, cfg, parent=None):
        obj = self.root.create(klass, cfg)
        obj.parent = parent
        obj.initialize(cfg)
        self.root.all_objects.append(obj)
        return obj

    def add_child(self, klass, cfg):
        obj = self.create_object(klass, cfg, parent=self)
        self.children.append(obj)
        return obj

    def create_shared_object(self, klass, uid, cfg):
        uid = _class_name(klass).replace('.', '_') + '_' + util.as_uid(uid)

        if uid in self.root.shared_objects:
            # log.debug(f'SHARED: FOUND {klass} {uid}')
            return self.root.shared_objects[uid]

        with util.global_lock:
            log.debug(f'SHARED: create {klass} {uid}')
            obj = self.create_object(klass, cfg)
            obj.uid = uid
            self.root.shared_objects[uid] = obj

        return obj

    def get_children(self, klass):
        return list(_find_all(self.children, klass))

    def get_closest(self, klass):
        if self.parent:
            if self.parent.is_a(klass):
                return self.parent
            return self.parent.get_closest(klass)

    def find_all(self, klass=None):
        return list(_find_all(self.root.all_objects, klass))

    def find_first(self, klass):
        for p in _find_all(self.root.all_objects, klass):
            return p

    def find(self, klass, uid):
        return _find(self.root.all_objects, klass, uid)

    def props_for(self, user):
        if not user.can_use(self):
            return None
        return _make_props(self.props, user)

    @property
    def props(self) -> t.Props:
        pass

#:stub
class RootBase(Object):
    def __init__(self):
        super().__init__()
        self.all_types = {}
        self.all_objects = []
        self.shared_objects = {}
        self.root = self

    def create(self, klass, cfg=None):
        oo = self._create(klass, cfg)
        oo.root = self
        return oo

    def _create(self, klass, cfg):
        if isinstance(klass, type):
            return klass()
        if cfg and cfg.get('type'):
            klass += '.' + cfg.get('type')
        if klass not in self.all_types:
            self.all_types[klass] = _load_class(klass)
        return self.all_types[klass]()


class ActionObject(Object):
    @property
    def props(self):
        return t.Props()


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
    if isinstance(obj, list):
        ls = []
        for v in obj:
            v = _make_props(v, user)
            if v is not None:
                ls.append(v)
        return ls

    if isinstance(obj, t.Data):
        obj = obj.as_dict()

    if isinstance(obj, dict):
        ls = {}
        for k, v in obj.items():
            v = _make_props(v, user)
            if v is not None:
                ls[k] = v
        return t.Props(ls)

    if isinstance(obj, Object):
        return obj.props_for(user)

    return obj
