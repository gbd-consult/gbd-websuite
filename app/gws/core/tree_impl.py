from . import (
    const as c,
    util as u,
    log,
)

Access = None
Error = None
ConfigurationError = None
Data = None
Props = None
Object = None


def object_repr(self):
    r = getattr(self, 'extName', None) or class_name(self)
    s = getattr(self, 'title', None)
    if s:
        r += f' title={s!r}'
    s = getattr(self, 'uid', None)
    if s:
        r += f' uid={s}'
    return '<' + r + ' ' + hex(id(self)) + '>'


def node_initialize(self, config):
    self.config = config
    self.permissions = configure_permissions(self)
    super_invoke(self, 'pre_configure')
    super_invoke(self, 'configure')


def node_create_child(self, classref, config, **kwargs):
    return self.root.create(classref, parent=self, config=config, **kwargs)


def node_create_child_if_configured(self, classref, config=None, **kwargs):
    if not config:
        return None
    return self.root.create(classref, parent=self, config=config, **kwargs)


def node_create_children(self, classref, configs, **kwargs):
    if not configs:
        return []
    return u.compact(self.create_child(classref, cfg, **kwargs) for cfg in configs)


def node_cfg(self, key: str, default=None):
    val = u.get(self.config, key)
    return val if val is not None else default


def node_find_all(self, classref):
    return find_all_in(self.root, self.children, classref)


def node_find_first(self, classref):
    return find_first_in(self.root, self.children, classref)


def node_find_closest(self, classref):
    node = self.parent
    while True:
        if not node or node is self.root:
            return
        if is_a(self.root, node, classref):
            return node
        node = node.parent


def node_find_ancestors(self, classref):
    pass


def node_find_descendants(self, classref):
    pass


def node_register_middleware(self, name: str, depends_on):
    self.root.app.middlewareMgr.register(self, name, depends_on)


##


def root_init(self, specs):
    self.specs = specs
    self.app = None
    self.permissions = {}
    self.configErrors = []
    self.configStack = []
    self.nodes = []
    self.uidMap = {}
    self.uidCount = 1


def root_initialize(self, node, config):
    self.configStack.append(node)

    try:
        node.initialize(config)
        ok = True
    except Exception as exc:
        log.exception()
        register_config_error(self, exc)
        ok = False

    self.configStack.pop()
    return ok


def root_post_initialize(self):
    for node in reversed(self.nodes):
        self.configStack = []
        p = node
        while p:
            self.configStack.insert(0, p)
            p = getattr(p, 'parent', None)
        try:
            super_invoke(node, 'post_configure')
        except Exception as exc:
            log.exception()
            register_config_error(self, exc)


def root_activate(self):
    for node in self.nodes:
        # if type(node).activate != Node.activate:
        #     log.debug(f'activate: {node!r}')
        node.activate()


def root_find_all(self, classref):
    return find_all_in(self, self.nodes, classref)


def root_find_first(self, classref):
    return find_first_in(self, self.nodes, classref)


def root_get(self, uid, classref):
    if not uid:
        return
    node = self.uidMap.get(uid)
    if node and (not classref or is_a(self, node, classref)):
        return node


def root_object_count(self) -> int:
    return len(self.nodes)


def root_create(self, classref, parent, config, **kwargs):
    config = to_config(config, kwargs)
    return create_node(self, classref, parent, config)


def root_create_shared(self, classref, config, **kwargs):
    config = to_config(config, kwargs)

    uid = config.uid
    if not uid:
        config.uid = '_s_' + u.sha256([repr(classref), config])

    if config.uid in self.uidMap:
        return self.uidMap[config.uid]

    return create_node(self, classref, None, config)


def root_create_temporary(self, classref, config, **kwargs):
    config = to_config(config, kwargs)
    return create_node(self, classref, None, config, temp=True)


def root_create_application(self, config, **kwargs):
    config = to_config(config, kwargs)

    node = alloc_node(self, 'gws.base.application.core.Object')
    node.uid = '1'
    node.parent = self
    node.children = []

    self.nodes.append(node)
    self.uidMap[node.uid] = node
    self.app = node

    self.initialize(node, config)

    return node


##

def class_name(node):
    return node.__class__.__module__ + '.' + node.__class__.__name__


def alloc_node(self, classref, typ=None):
    cls = self.specs.get_class(classref, typ)
    if not cls:
        raise Error(f'class {classref}:{typ} not found')

    node = cls()
    node.root = self
    node.extName = getattr(cls, 'extName', '')
    node.extType = getattr(cls, 'extType', '')

    return node


def configure_permissions(self):
    perms = {}

    p = self.cfg('access')
    if p:
        perms[Access.read] = u.parse_acl(p)

    p = self.cfg('permissions')
    if p:
        for k, v in vars(p).items():
            a = u.parse_acl(v)
            if a:
                if k == 'all':
                    perms[Access.read] = perms[Access.write] = perms[Access.create] = perms[Access.delete] = a
                elif k == 'edit':
                    perms[Access.write] = perms[Access.create] = perms[Access.delete] = a
                else:
                    perms[k] = a

    return perms


def create_node(self, classref, parent, config, temp=False):
    node = alloc_node(self, classref, config.get('type'))
    node.uid = get_or_generate_uid(self, config)
    node.parent = parent
    node.children = []

    log.debug('configure: ' + ('.' * 4 * len(self.configStack)) + f'{node!r} IN {parent or self !r}')
    ok = self.initialize(node, config)
    if not ok:
        log.debug(f'FAILED {node!r}')
        return

    if not temp:
        self.nodes.append(node)
        self.uidMap[node.uid] = node

    if parent:
        parent.children.append(node)

    return node


def find_all_in(root, where, classref):
    cls, name, ext_name = root.specs.parse_classref(classref)
    if cls:
        return [node for node in where if isinstance(node, cls)]
    if name:
        return [node for node in where if class_name(node) == name]
    if ext_name:
        return [node for node in where if node.extName.startswith(ext_name)]


def find_first_in(root, where, classref):
    ls = find_all_in(root, where, classref)
    return ls[0] if ls else None


def get_or_generate_uid(self, config):
    if config.get('uid'):
        return config.get('uid')
    self.uidCount += 1
    return str(self.uidCount)


def is_a(root, node, classref):
    cls, name, ext_name = root.specs.parse_classref(classref)
    if cls:
        return isinstance(node, cls)
    if name:
        return class_name(node) == name
    if ext_name:
        return node.extName.startswith(ext_name)
    return False


def props_of(node, user, *context):
    if not user.can_use(node, *context):
        return None
    p = make_props2(node, user)
    if p is None or isinstance(p, Data):
        return p
    if isinstance(p, dict):
        return Props(p)
    raise Error('invalid props type')


def make_props2(obj, user):
    if u.is_atom(obj):
        return obj

    if isinstance(obj, Object):
        if user.acl_bit(Access.read, obj) == c.DENY:
            return None
        obj = obj.props(user)

    if isinstance(obj, Data):
        obj = vars(obj)

    if u.is_dict(obj):
        return u.compact({k: make_props2(v, user) for k, v in obj.items()})

    if u.is_list(obj):
        return u.compact([make_props2(v, user) for v in obj])

    return None


def register_config_error(self, exc):
    lines = ['in ' + repr(val) for val in reversed(self.configStack)]
    err = ConfigurationError(repr(exc), *lines)
    err.__cause__ = exc
    self.configErrors.append(err)


def super_invoke(node, method):
    # since `super().configure` is mandatory in `configure` methods,
    # let's automate this by collecting all super 'configure' methods

    mro = []

    for cls in type(node).mro():
        try:
            if method in vars(cls):
                mro.append(cls)
        except TypeError:
            pass

    for cls in reversed(mro):
        getattr(cls, method)(node)


def to_config(config, defaults):
    return u.merge(Data(), defaults, config)
