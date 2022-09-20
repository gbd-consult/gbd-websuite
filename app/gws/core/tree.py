import sys

import gws
import gws.spec
from . import const, error, log, types, util, data

import gws.types as t


def class_name(obj):
    return obj.__class__.__module__ + '.' + obj.__class__.__name__


##


class Object(types.IObject):
    extName = ''
    extType = ''
    access = []
    uid = ''


class Node(Object, types.INode):
    def initialize(self, config):
        self.config = config
        self.access = self.var('access') or []

        # # since `super().configure` is mandatory in `configure` methods,
        # # let's automate this by collecting all super 'configure' methods upto 'Node'
        #
        # mro = []
        # for cls in type(self).mro():
        #     if cls == Node:
        #         break
        #     try:
        #         if 'configure' in vars(cls):
        #             mro.append(cls)
        #     except TypeError:
        #         pass
        #
        try:
            self.pre_configure()
            self.configure()
            # for cls in reversed(mro):
            #     cls.configure(self)  # type: ignore
            return True
        except Exception as exc:
            log.exception()
            self.root.configErrors.append(repr(exc))
            return False

    def var(self, key: str, default=None):
        val = util.get(self.config, key)
        return val if val is not None else default

    def create_child(self, classref, config=None, optional=False, required=False):
        return self.root.create(classref, parent=self, config=config, optional=optional, required=required)

    def create_children(self, classref, configs, required=False):
        if not configs:
            return []
        return gws.compact(self.create_child(classref, cfg, required=required) for cfg in configs)


class Root(types.IRoot):
    def __init__(self):
        self.app = t.cast(types.IApplication, None)
        self.configErrors = []
        self._cachedDescriptors: t.Dict[str, types.ExtObjectDescriptor] = {}
        self._objects: t.List['Node'] = []
        self._uidMap: t.Dict[str, 'Node'] = {}
        self._uidCount = 0

    def activate(self):
        for obj in self._objects:
            obj.activate()

    def post_initialize(self):
        for obj in reversed(self._objects):
            try:
                obj.post_configure()
            except Exception as exc:
                info = _error_info(exc)
                log.exception(info)
                self.configErrors.append(info)

    def find(self, classref, uid: str):
        obj = self._uidMap.get(uid)
        if not obj:
            return None
        cls, name, ext_name = self.specs.parse_classref(classref)
        if cls:
            return obj if isinstance(obj, cls) else None
        if name:
            return obj if class_name(obj) == name else None
        if ext_name:
            return obj if obj.extName.startswith(ext_name) else None

    def find_all(self, classref):
        cls, name, ext_name = self.specs.parse_classref(classref)
        if cls:
            return [obj for obj in self._objects if isinstance(obj, cls)]
        if name:
            return [obj for obj in self._objects if class_name(obj) == name]
        if ext_name:
            return [obj for obj in self._objects if obj.extName.startswith(ext_name)]

    def create_application(self, config=None):
        config = _to_config(config)

        cls = self.specs.get_class('gws.base.application.Object')

        obj = cls()
        obj.uid = '0'
        obj.root = self
        obj.parent = self
        obj.children = []

        self._objects.append(obj)
        self._uidMap[obj.uid] = obj
        self.app = obj
        obj.initialize(config)

    def create_shared(self, classref, config=None, optional=False, required=False):
        config = _to_config(config)
        if config.uid and config.uid in self._uidMap:
            return self._uidMap[config.uid]
        return self.create(classref, config=config, optional=optional, required=required)

    def create(self, classref, parent=None, config=None, optional=False, required=False):
        if not config and optional:
            return

        config = _to_config(config)

        config.uid = self._get_uid(config)
        cls = self.specs.get_class(classref, config.type)

        if not cls:
            raise error.ConfigurationError(f'class {classref!r}:{config.type!r} not found')

        obj = cls()
        obj.root = self
        obj.parent = parent
        obj.children = []

        log.debug(f'configuring {_object_name(obj)}')
        ok = obj.initialize(config)
        if not ok:
            log.debug(f'FAILED {_object_name(obj)}')
            return

        if not obj.uid:
            obj.uid = config.uid
        self._objects.append(obj)
        self._uidMap[obj.uid] = obj

        if parent:
            parent.children.append(obj)

        return obj


    def _get_uid(self, config):
        if config.uid:
            return config.uid
        self._uidCount += 1
        return str(self._uidCount)


##


def create_root_object(specs: types.ISpecRuntime) -> types.IRoot:
    r = Root()
    r.specs = specs
    return r


##


def _to_config(config):
    if util.is_data_object(config):
        return config
    if isinstance(config, dict):
        return types.Config(config)
    if not config:
        return types.Config()
    return config


def _object_name(obj):
    name = util.get(obj, 'extName', '') or class_name(obj)
    uid = util.get(obj, 'uid', '')
    if uid:
        name += ' (uid=' + str(uid) + ')'
    return name




##


def props(obj: types.IObject, user: types.IGrantee, context: t.Optional[types.INode] = None) -> t.Optional[types.Props]:
    if not user.can_use(obj, context):
        return None
    p = _make_props(obj.props(user), user)
    if p is None or util.is_data_object(p):
        return p
    if isinstance(p, dict):
        return types.Props(p)
    raise error.Error('invalid props type')


def _make_props(obj: t.Any, user):
    if util.is_atom(obj):
        return obj

    if user.access_to(obj) == gws.ACCESS_DENIED:
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
