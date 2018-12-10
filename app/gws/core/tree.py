import importlib

from . import util, error, log
import gws.types as t


class Object(t.ObjectInterface):
    def __init__(self):
        self.children = []
        self.config = None
        self.parent = None
        self.root: 'RootObject' = None
        self.uid = ''

        self.access = None
        self.klass = _class_name(self.__class__)

    def is_a(self, klass):
        if isinstance(klass, type):
            return isinstance(self, klass)
        if self.klass == klass:
            return True
        return self.klass.startswith(klass + '.')

    def initialize(self, cfg):
        self.config = cfg
        self.uid = _get_uid(cfg, self.klass)
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

    def find_all(self, klass):
        return list(_find_all(self.root.all_objects, klass))

    def find_first(self, klass):
        for p in _find_all(self.root.all_objects, klass):
            return p

    def find(self, klass, uid):
        return _find(self.root.all_objects, klass, uid)


class RootObject(Object):
    def __init__(self):
        super().__init__()
        self.all_types = {}
        self.all_objects = []
        self.shared_objects = {}
        self.root = self

    def load_class(self, klass):
        try:
            mod = importlib.import_module(klass)
        except Exception as e:
            raise error.Error('import of %r failed' % klass) from e
        try:
            return mod.Object
        except Exception as e:
            raise error.Error('object not found in %r' % klass) from e

    def create(self, klass, cfg=None):
        if isinstance(klass, type):
            oo = klass()
        else:
            if cfg and cfg.get('type'):
                klass += '.' + cfg.get('type')
            if klass not in self.all_types:
                self.all_types[klass] = self.load_class(klass)
            oo = self.all_types[klass]()

        oo.root = self
        return oo


class PublicObject(Object):
    def props_for(self, user):
        if not user.can_read(self):
            return None
        return _make_props(self.props, user)

    @property
    def props(self):
        return None


def _find(nodes, klass, uid):
    if not uid:
        return
    for obj in nodes:
        if obj.uid == uid and obj.is_a(klass):
            return obj


def _find_all(nodes, klass):
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


_object_count = 0


def _get_uid(cfg, klass):
    global _object_count

    u = cfg.get('uid')
    if u:
        return util.as_uid(u)

    u = cfg.get('title')
    if u:
        return util.as_uid(u)

    u = cfg.get('type')
    _object_count += 1
    return str(klass).replace('.', '_') + '_' + str(_object_count)


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
        p = [_make_props(v, user) for v in obj]
        return [v for v in p if v is not None]

    if isinstance(obj, dict):
        p = {k: _make_props(v, user) for k, v in obj.items()}
        return {k: v for k, v in p.items() if v is not None}

    if isinstance(obj, PublicObject):
        return obj.props_for(user)

    return obj
