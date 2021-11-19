import sys

import gws
import gws.types as t

from . import const, error, log, types, util

_ALLOW = 'allow'
_DENY = 'deny'

PUBLIC = 'all:allow'


class Object(types.IObject):
    def __init__(self):
        _create_attributes(self)
        self.class_name = _class_name(self)
        self.access = None
        self.is_shared = False

    def props_for(self, user):
        return types.Props()

    def access_for(self, user):
        if self.access:
            for a in self.access:
                if a.role in user.roles:
                    return a.type == _ALLOW

    def is_a(self, klass: types.Klass):
        if isinstance(klass, type):
            return isinstance(self, klass)
        return self.class_name == klass or self.class_name.startswith(klass + '.')


##

class Node(Object, types.INode):
    def __init__(self):
        super().__init__()
        self.children = []

    def is_a(self, klass: types.Klass):
        if isinstance(klass, type):
            return isinstance(self, klass)
        return self.root.specs.is_a(self.class_name, klass)

    ##

    def configure(self):
        pass

    def post_configure(self):
        pass

    def activate(self):
        pass

    def initialize(self, cfg):
        self.config = cfg

        a = self.var('access')
        if isinstance(a, str):
            # access in the sting format `role:type,role:type...`
            self.access = []
            for s in a.lower().split(','):
                r, t = s.split(':')
                self.access.append(gws.Access(role=r.strip(), type=t.strip()))
        else:
            self.access = a

        self.set_uid()

        log.debug(f'BEGIN config {self.class_name}')

        # since `super().configure` is mandatory in `configure` methods,
        # let's automate this by collecting all super 'configure' methods upto 'Node'

        mro = []
        for cls in type(self).mro():
            if cls == Node:
                break
            try:
                if 'configure' in vars(cls):
                    mro.append(cls)
            except TypeError:
                pass

        try:
            for cls in reversed(mro):
                cls.configure(self)  # type: ignore
            log.debug(f'END config {self.class_name} uid={self.uid}')
            return True
        except Exception as exc:
            info = _error_info(exc, self)
            self.root.configuration_errors.append(info)
            log.exception(info.replace('\n', ' '))
            log.debug(f'FAILED config {self.class_name}')
            return False

    def set_uid(self, uid: str = ''):
        self.root.set_object_uid(self, uid)

    def var(self, key: str, default=None, with_parent=False):
        v = util.get(self.config, key)
        if v is not None:
            return v
        if with_parent and self.parent:
            return self.parent.var(key, default, with_parent=True)
        return default

    def create_child(self, klass, cfg):
        return self.root.create_object(klass, cfg, parent=self)

    def create_child_if_config(self, klass, cfg):
        if cfg:
            return self.create_child(klass, cfg)
        return None

    def require_child(self, klass, cfg):
        obj = self.create_child(klass, cfg)
        if not obj:
            raise error.Error(f'{klass!r} failed to initialize')
        return obj

    def create_children(self, klass, cfgs):
        if not cfgs:
            return []
        ls = []
        for cfg in cfgs:
            obj = self.create_child(klass, cfg)
            if obj:
                ls.append(obj)
        return ls

    def get_closest(self, klass):
        obj = self.parent
        while obj:
            if obj.is_a(klass):
                return obj
            obj = obj.parent
        return None


##


class Root(types.IRoot):

    def __init__(self):
        super().__init__()

        self.specs = ...  # to be populated in create_root_object
        self.root = self
        self.configuration_errors = []

        self._descriptors: t.Dict[str, types.ExtObjectDescriptor] = {}
        self._objects: t.List['Node'] = []
        self._shared_objects: t.Dict[str, 'Node'] = {}
        self._uids = set()

    def set_object_uid(self, obj, uid=None):

        def _auto():
            u = obj.var('uid')
            if u:
                return u
            return obj.class_name.replace('.', '_')

        def _new(uid):
            n = 0
            u = uid
            while u in self._uids:
                n += 1
                u = uid + str(n)
            self._uids.add(u)
            return u

        if obj.uid and obj.uid == uid:
            return

        uid = uid or _auto()

        with util.app_lock():
            if obj.uid:
                self._uids.discard(obj.uid)
            obj.uid = _new(uid)

    def post_initialize(self):
        for obj in reversed(self._objects):
            try:
                obj.post_configure()
            except Exception as exc:
                info = _error_info(exc, self)
                self.configuration_errors.append(info)
                log.exception(info.replace('\n', ' '))

    def activate(self):
        for obj in self._objects:
            obj.activate()

    def create_application(self, cfg):
        cfg = _to_config(cfg)
        app = self._create_from_klass_and_config('gws.base.application', _to_config({}))
        self._objects.append(app)
        self.application = app
        t.cast('Node', self.application).initialize(cfg)

    def create_object(self, klass, cfg=None, parent=None, shared=False, key=None):
        cfg = _to_config(cfg)

        if not shared:
            return self._create_and_initialize(klass, cfg, parent)

        if not key:
            key = util.sha256(cfg)
        elif isinstance(key, str):
            key = util.to_uid(key)
        else:
            key = util.sha256(key)

        key = util.to_uid(_class_name(klass)) + '_' + key

        if key in self._shared_objects:
            return self._shared_objects[key]

        with util.app_lock():
            log.debug(f'create_shared_object: klass={klass!r} key={key!r}')
            obj = self._create_and_initialize(klass, cfg, parent)
            setattr(obj, 'is_shared', True)
            if obj:
                self._shared_objects[key] = obj

        return obj

    ##

    def _create_and_initialize(self, klass: types.Klass, cfg, parent: 'Node' = None) -> t.Optional['Node']:
        obj = self._create_from_klass_and_config(klass, cfg)
        if parent:
            obj.parent = parent
        if not obj.initialize(cfg):
            return None
        if parent:
            parent.children.append(obj)
        self._objects.append(obj)
        return obj

    def _create_from_klass_and_config(self, klass: types.Klass, cfg):
        desc = self._find_descriptor(klass, cfg)
        obj = desc.class_ptr()
        obj.root = self
        obj.ext_category = desc.ext_category
        obj.ext_type = desc.ext_type
        return obj

    def _find_descriptor(self, klass: types.Klass, cfg):
        class_name = _class_name(klass) if isinstance(klass, type) else klass

        cs = class_name.split('.')
        if not cs[-1][0].isupper():
            # not a fully qualified name
            if class_name.startswith('gws.ext') and cfg.get('type'):
                # e.g. 'gws.ext.action' + type=auth -> gws.ext.action.auth.Object
                cs.append(cfg.get('type'))
            cs.append('Object')
            class_name = '.'.join(cs)

        if class_name in self._descriptors:
            return self._descriptors[class_name]
        if class_name in _ADHOC_TYPES:
            return _ADHOC_TYPES[class_name]

        rcs = self.specs.real_class_names(class_name)
        if len(rcs) != 1:
            raise error.Error(f'_find_descriptor: invalid class {class_name!r} (found {rcs!r})')

        desc = self.specs.object_descriptor(rcs[0])
        if not desc:
            raise error.Error(f'_find_descriptor: class {class_name!r} ({rcs!r}) not found')

        if not desc.class_ptr:
            if desc.module_name in sys.modules:
                log.debug(f'_find_descriptor: {class_name!r}: already loaded')
                mod = sys.modules[desc.module_name]
            else:
                log.debug(f'_find_descriptor: {class_name!r}: loading from spec: {desc!r}')
                mod = util.import_from_path(desc.module_path)
            desc.class_ptr = getattr(mod, desc.ident)

        self._descriptors[class_name] = desc
        return desc

    def find_all(self, klass=None, uid=None):
        ls = self._objects
        if klass:
            ls = [obj for obj in ls if obj.is_a(klass)]
        if uid:
            ls = [obj for obj in ls if obj.uid == uid]
        return ls

    def find(self, klass=None, uid=None):
        ls = self.find_all(klass, uid)
        return ls[0] if ls else None


##

def create_root_object(specs: types.ISpecRuntime):
    r = Root()
    r.specs = specs
    return r


##

_ADHOC_TYPES = {}


def register_ext(class_name: str, cls: type):
    _ADHOC_TYPES[class_name] = types.ExtObjectDescriptor(
        name=class_name,
        ext_type=class_name.split('.')[-2],  # class_name is like "gws.ext.auth.provider.mock.Node"
        class_ptr=cls)


def unregister_ext():
    _ADHOC_TYPES.clear()


##

_ACCCESS_ALLOWED = 1
_ACCCESS_DENIED = 2
_ACCCESS_UNKNOWN = 3


def user_can_use(user, obj, context=None):
    return _access(user, obj, context) == _ACCCESS_ALLOWED


def _access(user, obj, context):
    def _log(res, why, where=None):
        # log.debug(f'PERMS:{res}: o={_object_name(obj)} r={user.roles!r}: why={why} where={_object_name(where)}')
        pass

    if not obj:
        _log(False, 'empty')
        return _ACCCESS_DENIED

    if obj == user:
        _log(True, 'self')
        return _ACCCESS_ALLOWED

    if const.ROLE_ADMIN in user.roles:
        _log(True, 'admin')
        return _ACCCESS_ALLOWED

    fn = getattr(obj, 'access_for', None)
    c = fn(user) if fn else None

    if c is not None:
        _log(c, 'found')
        return _ACCCESS_ALLOWED if c else _ACCCESS_DENIED

    curr = context or util.get(obj, 'parent')
    if not curr and getattr(obj, 'is_shared', False):
        raise error.Error(f'no access context found for {_object_name(obj)}')

    while curr:
        fn = getattr(curr, 'access_for', None)
        c = fn(user) if fn else None
        if c is not None:
            _log(c, 'found', curr)
            return _ACCCESS_ALLOWED if c else _ACCCESS_DENIED

        curr = util.get(curr, 'parent')

    _log(None, 'end')
    return _ACCCESS_UNKNOWN


_all_user = t.cast(types.IGrantee, types.Data(roles={const.ROLE_ALL}))


def is_public_object(obj):
    return _access(_all_user, obj, None) == _ACCCESS_ALLOWED


##


def props(obj: types.IObject, user: types.IGrantee, context: t.Optional[types.INode] = None) -> t.Optional[types.Props]:
    if not user.can_use(obj, context):
        return None
    p = _make_props(obj.props_for(user), user)
    if p is None or util.is_data_object(p):
        return p
    if isinstance(p, dict):
        return types.Props(p)
    raise error.Error('invalid props type')


def _make_props(obj: t.Any, user):
    if util.is_atom(obj):
        return obj

    if isinstance(obj, Object):
        if _access(user, obj, None) == _ACCCESS_DENIED:
            return None
        obj = obj.props_for(user)

    if util.is_data_object(obj):
        obj = vars(obj)

    if util.is_dict(obj):
        return util.compact({k: _make_props(v, user) for k, v in obj.items()})

    if util.is_list(obj):
        return util.compact([_make_props(v, user) for v in obj])

    return None


##

def _attribute_names_for(obj):
    names = set()
    cls = type(obj)

    for c in cls.mro():
        ann = getattr(c, '__annotations__', None)
        if ann:
            names.update(ann)

    return set(n for n in names if not hasattr(cls, n))


def _create_attributes(obj):
    for n in _attribute_names_for(obj):
        setattr(obj, n, None)


def _to_config(cfg):
    if util.is_data_object(cfg):
        return cfg
    if isinstance(cfg, dict):
        return types.Config(cfg)
    if not cfg:
        return types.Config()
    return cfg


def _class_name(s):
    if isinstance(s, str):
        return s
    if not isinstance(s, type):
        s = type(s)
    m = getattr(s, '__module__', '__unknown_module')
    n = getattr(s, '__name__', '__unknown_name')
    return m + '.' + n


def _object_name(obj):
    name = getattr(obj, 'class_name', '')
    if not name:
        try:
            name = obj.__class__.__name__
        except:
            name = 'object'

    uid = getattr(obj, 'uid', '')
    if uid:
        name += ' (' + str(uid) + ')'

    return name


def _error_info(exc, obj):
    try:
        cls = exc.__class__.__name__
    except:
        cls = 'Error'

    try:
        info = cls + ': ' + exc.args[0]
    except:
        info = cls

    stack = ''

    while obj:
        name = _object_name(obj)
        stack += 'in ' + name + '\n'
        obj = getattr(obj, 'parent', None)

    if stack:
        info += '\n' + stack.strip()

    return info
