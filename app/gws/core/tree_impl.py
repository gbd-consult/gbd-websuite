"""Internal implementations of ``gws.Node`` and ``gws.Root`` methods."""

from . import (
    const as c,
    util as u,
    log,
)

Access = None
Error = None
Data = None
Props = None
Object = None


def object_repr(self):
    """Return a short representation of an object.

    Args:
        self: The object.

    Returns:
        A string with the class or extension name, title, uid and the object id.
    """
    r = getattr(self, 'extName', None) or class_name(self)
    s = getattr(self, 'title', None)
    if s:
        r += f' title={s!r}'
    s = getattr(self, 'uid', None)
    if s:
        r += f' uid={s}'
    return '<' + r + ' ' + hex(id(self)) + '>'


def node_initialize(self, config):
    """Implement ``gws.Node.initialize``.

    Args:
        self: The node.
        config: Configuration.
    """
    self.config = config
    self.permissions = configure_permissions(self)
    super_invoke(self, 'pre_configure')
    super_invoke(self, 'configure')


def node_create_child(self, classref, config, **kwargs):
    """Implement ``gws.Node.create_child``.

    Args:
        self: The parent node.
        classref: Class reference.
        config: Configuration.
        **kwargs: Additional configuration properties.

    Returns:
        A newly created node or ``None``.
    """
    return self.root.create(classref, parent=self, config=config, **kwargs)


def node_create_child_if_configured(self, classref, config=None, **kwargs):
    """Implement ``gws.Node.create_child_if_configured``.

    Args:
        self: The parent node.
        classref: Class reference.
        config: Configuration.
        **kwargs: Additional configuration properties.

    Returns:
        A newly created node or ``None``.
    """
    if not config:
        return None
    return self.root.create(classref, parent=self, config=config, **kwargs)


def node_create_children(self, classref, configs, **kwargs):
    """Implement ``gws.Node.create_children``.

    Args:
        self: The parent node.
        classref: Class reference.
        configs: List of configurations.
        **kwargs: Additional configuration properties.

    Returns:
        A list of newly created nodes, without the ones that failed.
    """
    if not configs:
        return []
    return u.compact(self.create_child(classref, cfg, **kwargs) for cfg in configs)


def node_cfg(self, key: str, default=None):
    """Implement ``gws.Node.cfg``.

    Args:
        self: The node.
        key: Property key, nested keys are separated by dots.
        default: Value to return if the property is ``None`` or missing.

    Returns:
        The property value or the default.
    """
    val = u.get(self.config, key)
    return val if val is not None else default


def node_find_all(self, classref):
    """Implement ``gws.Node.find_all``.

    Args:
        self: The node.
        classref: Class reference. If ``None``, all nodes match.

    Returns:
        A list of matching children.
    """
    return find_all_in(self.root, self.children, classref)


def node_find_first(self, classref):
    """Implement ``gws.Node.find_first``.

    Args:
        self: The node.
        classref: Class reference. If ``None``, all nodes match.

    Returns:
        The first matching child or ``None``.
    """
    return find_first_in(self.root, self.children, classref)


def node_find_closest(self, classref):
    """Implement ``gws.Node.find_closest``.

    Args:
        self: The node.
        classref: Class reference. If ``None``, all nodes match.

    Returns:
        The closest matching ancestor below the root, or ``None``.
    """
    node = self.parent
    while True:
        if not node or node is self.root:
            return
        if not classref or is_a(self.root, node, classref):
            return node
        node = node.parent


def node_find_ancestors(self, classref):
    """Implement ``gws.Node.find_ancestors``.

    Args:
        self: The node.
        classref: Class reference. If ``None``, all nodes match.

    Returns:
        A list of matching ancestors below the root, from the parent upwards.
    """
    ls = []
    node = self.parent

    while True:
        if not node or node is self.root:
            break
        if not classref or is_a(self.root, node, classref):
            ls.append(node)
        node = node.parent

    return ls


def node_find_descendants(self, classref):
    """Implement ``gws.Node.find_descendants``.

    Args:
        self: The node.
        classref: Class reference. If ``None``, all nodes match.

    Returns:
        A list of matching descendants in the depth-first order.
    """
    ls = []

    def _walk(node):
        for child_node in node.children:
            if not classref or is_a(self.root, child_node, classref):
                ls.append(child_node)
            _walk(child_node)

    _walk(self)
    return ls


##


def root_init(self, specs):
    """Implement ``gws.Root.__init__``.

    Args:
        self: The root.
        specs: Specs runtime.
    """
    self.specs = specs
    self.app = None
    self.permissions = {}
    self.configErrors = []
    self.configWarnings = []
    self.configStack = []
    self.configPaths = []
    self.nodes = []
    self.uidMap = {}
    self.uidCount = 1


def root_initialize(self, node, config):
    """Implement ``gws.Root.initialize``.

    The node is pushed onto ``configStack`` while it is initialized.
    Exceptions are logged and recorded in ``configErrors``.

    Args:
        self: The root.
        node: The node.
        config: Configuration.

    Returns:
        ``True`` if the node was initialized, ``False`` if it failed.
    """
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
    """Implement ``gws.Root.post_initialize``.

    Args:
        self: The root.
    """
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
    self.configStack = []


def root_activate(self):
    """Implement ``gws.Root.activate``.

    Args:
        self: The root.
    """
    for node in self.nodes:
        # if type(node).activate != Node.activate:
        #     log.debug(f'activate: {node!r}')
        node.activate()


def root_find_all(self, classref):
    """Implement ``gws.Root.find_all``.

    Args:
        self: The root.
        classref: Class reference. If ``None``, all nodes match.

    Returns:
        A list of matching nodes.
    """
    return find_all_in(self, self.nodes, classref)


def root_find_first(self, classref):
    """Implement ``gws.Root.find_first``.

    Args:
        self: The root.
        classref: Class reference. If ``None``, all nodes match.

    Returns:
        The first matching node or ``None``.
    """
    return find_first_in(self, self.nodes, classref)


def root_get(self, uid, classref):
    """Implement ``gws.Root.get``.

    Args:
        self: The root.
        uid: Node uid.
        classref: Class reference. If given, the node must match it.

    Returns:
        The node or ``None``.
    """
    if not uid:
        return
    node = self.uidMap.get(uid)
    if node and (not classref or is_a(self, node, classref)):
        return node


def root_object_count(self) -> int:
    """Implement ``gws.Root.object_count``.

    Args:
        self: The root.

    Returns:
        The number of nodes.
    """
    return len(self.nodes)


def root_create(self, classref, parent, config, **kwargs):
    """Implement ``gws.Root.create``.

    Args:
        self: The root.
        classref: Class reference.
        parent: Parent node.
        config: Configuration.
        **kwargs: Additional configuration properties.

    Returns:
        A newly created node or ``None``.
    """
    config = to_config(config, kwargs)
    return create_node(self, classref, parent, config)


def root_create_shared(self, classref, config, **kwargs):
    """Implement ``gws.Root.create_shared``.

    Args:
        self: The root.
        classref: Class reference.
        config: Configuration.
        **kwargs: Additional configuration properties.

    Returns:
        An existing node with the same uid, a newly created node, or ``None``.
    """
    config = to_config(config, kwargs)

    uid = config.uid
    if not uid:
        config.uid = '_s_' + u.sha256([repr(classref), config])

    if config.uid in self.uidMap:
        return self.uidMap[config.uid]

    return create_node(self, classref, None, config)


def root_create_temporary(self, classref, config, **kwargs):
    """Implement ``gws.Root.create_temporary``.

    The node is not registered in the root and ``post_configure`` is run immediately.

    Args:
        self: The root.
        classref: Class reference.
        config: Configuration.
        **kwargs: Additional configuration properties.

    Returns:
        A newly created node or ``None``.
    """
    config = to_config(config, kwargs)
    node = create_node(self, classref, None, config, temp=True)
    if node:
        super_invoke(node, 'post_configure')
    return node


def root_create_application(self, config, **kwargs):
    """Implement ``gws.Root.create_application``.

    The application gets the fixed uid ``const.APPLICATION_UID``.

    Args:
        self: The root.
        config: Configuration.
        **kwargs: Additional configuration properties.

    Returns:
        The Application object.
    """
    config = to_config(config, kwargs)

    node = alloc_node(self, 'gws.base.application.core.Object')
    node.uid = c.APPLICATION_UID
    node.parent = self
    node.children = []

    self.nodes.append(node)
    self.uidMap[node.uid] = node
    self.app = node

    self.initialize(node, config)

    return node


##


def class_name(node):
    """Return the full class name of an object.

    Args:
        node: An object.

    Returns:
        The module and class name, like ``gws.base.layer.core.Object``.
    """
    return node.__class__.__module__ + '.' + node.__class__.__name__


def alloc_node(self, classref, typ=None):
    """Create an uninitialized node of the class found in the specs.

    Args:
        self: The root.
        classref: Class reference.
        typ: Extension type, for ``ext`` class references.

    Returns:
        The new node, with ``root``, ``extName`` and ``extType`` set.

    Raises:
        Error: If the class is not found.
    """
    cls = self.specs.get_class(classref, typ)
    if not cls:
        raise Error(f'class {classref}:{typ} not found')

    node = cls()
    node.root = self
    node.extName = getattr(cls, 'extName', '')
    node.extType = getattr(cls, 'extType', '')

    return node


def configure_permissions(self):
    """Compute the permissions of a node from its ``access`` and ``permissions`` config.

    ``access`` sets the read permission. In ``permissions``, ``all`` sets all modes, ``edit`` sets write,
    create and delete, and the specific modes override both.

    Args:
        self: The node.

    Returns:
        A mapping from an access mode to an ACL.
    """
    perms = {
        Access.read: [],
        Access.write: [],
        Access.create: [],
        Access.delete: [],
    }

    p = self.cfg('access')
    if p:
        perms[Access.read] = u.parse_acl(p)

    p = self.cfg('permissions')
    if p:
        if isinstance(p, Data):
            p = vars(p)
        v = p.get('all')
        if v:
            perms[Access.read] = perms[Access.write] = perms[Access.create] = perms[Access.delete] = u.parse_acl(v)

        v = p.get('edit')
        if v:
            perms[Access.write] = perms[Access.create] = perms[Access.delete] = u.parse_acl(v)

        for k in {Access.read, Access.write, Access.create, Access.delete}:
            v = p.get(k)
            if v:
                perms[k] = u.parse_acl(v)

    return perms


def create_node(self, classref, parent, config, temp=False):
    """Create and initialize a node.

    Args:
        self: The root.
        classref: Class reference.
        parent: Parent node, the new node is appended to its children.
        config: Configuration. Its ``type`` selects the extension type.
        temp: If ``True``, the node is not registered in the root.

    Returns:
        The new node, or ``None`` if the initialization failed.
    """
    node = alloc_node(self, classref, config.get('type'))
    node.uid = get_or_generate_uid(self, config)
    node.parent = parent
    node.children = []

    log.debug('configure: ' + ('.' * 4 * len(self.configStack)) + f'{node!r} IN {parent or self!r}')
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


def find_all_in(root, nodes, classref):
    """Filter nodes by a class reference.

    Args:
        root: The root.
        nodes: A list of nodes.
        classref: Class reference. If ``None``, all nodes match.

    Returns:
        A list of matching nodes.
    """
    if not classref:
        return nodes
    cls, name, ext_name = root.specs.parse_classref(classref)
    if cls:
        return [node for node in nodes if isinstance(node, cls)]
    if name:
        return [node for node in nodes if class_name(node) == name]
    if ext_name:
        return [node for node in nodes if node.extName.startswith(ext_name)]


def find_first_in(root, nodes, classref):
    """Find the first node that matches a class reference.

    Args:
        root: The root.
        nodes: A list of nodes.
        classref: Class reference. If ``None``, all nodes match.

    Returns:
        The first matching node or ``None``.
    """
    found = find_all_in(root, nodes, classref)
    return found[0] if found else None


def get_or_generate_uid(self, config):
    """Return the uid from the config, or generate a new numeric one.

    Args:
        self: The root.
        config: Configuration.

    Returns:
        A uid.
    """
    if config.get('uid'):
        return config.get('uid')
    self.uidCount += 1
    return str(self.uidCount)


def is_a(root, node, classref):
    """Check if a node matches a class reference.

    A class matches by ``isinstance``, a class name matches exactly,
    an ``ext`` name matches the node's extension name or its prefix.

    Args:
        root: The root.
        node: The node.
        classref: Class reference.

    Returns:
        ``True`` if the node matches.
    """
    cls, name, ext_name = root.specs.parse_classref(classref)
    if cls:
        return isinstance(node, cls)
    if name:
        return class_name(node) == name
    if ext_name:
        return node.extName == ext_name or node.extName.startswith(ext_name + '.')
    return False


def props_of(node, user, *context):
    """Implement ``gws.props_of``.

    Args:
        node: The object.
        user: The user.
        *context: Context objects for the permission check.

    Returns:
        A ``Props`` object or ``None``.

    Raises:
        Error: If the object's ``props`` returns an invalid type.
    """
    if not user.can_use(node, *context):
        return None
    p = make_props2(node, user)
    if p is None or isinstance(p, Data):
        return p
    if isinstance(p, dict):
        return Props(p)
    raise Error('invalid props type')


def make_props2(obj, user):
    """Recursively convert a value to props for a user.

    Objects are converted with their ``props`` method, or dropped if the user has no read permission.
    ``None`` values are removed from dicts and lists.

    Args:
        obj: An object, ``Data``, dict, list or a scalar value.
        user: The user.

    Returns:
        A scalar, dict or list, or ``None``.
    """
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
    """Record an exception in ``configErrors``, with the current config stack.

    Args:
        self: The root.
        exc: The exception.
    """
    try:
        msg = getattr(exc, 'message', None) or str(exc.args[0])
    except:
        msg = repr(exc)
    self.configErrors.append(config_info(self, msg))


def root_config_warning(self, message):
    """Implement ``gws.Root.config_warning``.

    Args:
        self: The root.
        message: Warning message.
    """
    cei = config_info(self, message)
    loc = ''
    if cei.stack:
        loc = ' in ' + config_location_repr(cei.stack[0])
    log.warning(f'CONFIGURATION WARNING: {message}{loc}')
    self.configWarnings.append(cei)


def config_info(self, message):
    """Create a configuration error or warning record.

    Args:
        self: The root.
        message: The message.

    Returns:
        A ``Data`` object with ``message`` and ``stack``, a list of locations from the innermost node outwards.
    """
    cei = Data(message=message, stack=[])
    for node in reversed(self.configStack):
        cei.stack.append(
            # @TODO actually this is a ConfigLocation object
            Data(
                objectUid=getattr(node, 'uid', ''),
                objectType=getattr(node, 'extName', None) or class_name(node),
                objectName=getattr(node, 'name', '') or getattr(node, 'title', ''),
                propName='',
            )
        )
    return cei


def config_location_repr(loc):
    """Return a short representation of a configuration location.

    Args:
        loc: A location from ``config_info``.

    Returns:
        A string with the object type, name, uid and the property name.
    """
    p = [
        loc.objectType,
        repr(loc.objectName) if loc.objectName else None,
        f'uid={loc.objectUid}' if loc.objectUid else None,
    ]
    p = '<' + ' '.join(u.compact(p)) + '>'
    if loc.propName:
        p = f'{loc.propName!r} {p}'
    return p


def super_invoke(node, method):
    """Invoke a method of every class in the node's MRO that defines it, base classes first.

    Args:
        node: The node.
        method: Method name.
    """
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
    """Merge keyword defaults and a configuration into a new ``Data`` object.

    Values in ``config`` override the defaults, ``None`` values are skipped.

    Args:
        config: Configuration.
        defaults: Default values.

    Returns:
        A new ``Data`` object.
    """
    return u.merge(Data(), defaults, config)
