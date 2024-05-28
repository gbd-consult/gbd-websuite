"""Basic types.

This module contains essential type definitions and utilities from the core GWS library.
It should be imported in every gws module.
"""

from typing import (
    TYPE_CHECKING,
    TypeAlias,
    cast,
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Union,
)

from collections.abc import (
    Mapping,
    Sequence,
)

import enum

if TYPE_CHECKING:
    import datetime
    import sqlalchemy
    import sqlalchemy.orm
    import numpy.typing

# mypy: disable-error-code="empty-body"


from . import ext

from .core import (
    log,
    debug,
    env,
    const as c,
    util as u,
)


################################################################################
# /core/_data.pyinc


# basic data type

class Data:
    """Basic data object.

    This object can be instantiated by passing one or or ``dict`` arguments
    and/or keyword args. All dicts keys and keywords become attributes of the object.

    Accessing an undefined attribute returns ``None`` and no error is raised,
    unless the attribute name starts with an underscore.
    """

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __repr__(self):
        return repr(vars(self))

    def __getitem__(self, key):
        return vars(self).get(key)

    def __setitem__(self, key, value):
        vars(self)[key] = value

    def get(self, key, default=None):
        """Get an attribute value.

        Args:
            key: Attribute name.
            default: Default value, returned if the attribute is undefined.
        """
        return vars(self).get(key, default)

    def setdefault(self, key, val):
        """Set an attribute value if not already set.

        Args:
            key: Attribute name.
            val: Attribute value.
        """
        return vars(self).setdefault(key, val)

    def set(self, key, val):
        """Set an attribute value.

        Args:
            key: Attribute name.
            val: Attribute value.
        """
        vars(self)[key] = val

    def update(self, *args, **kwargs):
        """Update the object with keys and values from args and keywords.

        Args:
            *args: Dicts or Mappings.
            kwargs: Keyword args.
        """

        d = {}
        for a in args:
            if isinstance(a, Mapping):
                d.update(a)
            elif isinstance(a, Data):
                d.update(vars(a))
        d.update(kwargs)
        vars(self).update(d)


# getattr needs to be defined out of class, otherwise IDEA accepts all attributes

def _data_getattr(self, attr):
    if attr.startswith('_'):
        # do not use None fallback for special props
        raise AttributeError(attr)
    return None


setattr(Data, '__getattr__', _data_getattr)


def is_data_object(x):
    """True if the argument is a ``Data`` object."""
    return isinstance(x, Data)


def to_data_object(x) -> 'Data':
    """Convert a value to a ``Data`` object.

    If the argument is already a ``Data`` object, simply return it.
    If the argument is ``None``, an empty object is returned.

    Args:
        x: A Mapping or ``None``.
    """

    if is_data_object(x):
        return x
    if isinstance(x, Mapping):
        return Data(x)
    if x is None:
        return Data()
    raise ValueError(f'cannot convert {x!r} to Data')
################################################################################



u.is_data_object = is_data_object
u.to_data_object = to_data_object



################################################################################
# /core/_basic.pyinc


class Enum(enum.Enum):
    """Enumeration type.

    Despite being declared as extending ``Enum`` (for IDE support), this class is actually just a simple object
    and intended to be used as a collection of attributes. It doesn't provide any ``Enum``-specific utilities.

    The rationale behind this is that we need ``Enum`` members (e.g. ``Color.RED``) to be scalars,
    and not complex objects as in the standard ``Enum``.
    """
    pass


# hack to make Enum a simple object
globals()['Enum'] = type('Enum', (), {})

Extent: TypeAlias = tuple[float, float, float, float]
"""An array of 4 elements representing extent coordinates ``[min-x, min-y, max-x, max-y]``."""

Point: TypeAlias = tuple[float, float]
"""Point coordinates ``[x, y]``."""

Size: TypeAlias = tuple[float, float]
"""Size ``[width, height]``."""


class Origin(Enum):
    """Grid origin."""

    nw = 'nw'
    """north-west"""
    sw = 'sw'
    """south-west"""
    ne = 'ne'
    """north-east"""
    se = 'se'
    """south-east"""
    lt = 'nw'
    """left top"""
    lb = 'sw'
    """left bottom"""
    rt = 'ne'
    """right top"""
    rb = 'se'
    """right bottom"""


FilePath: TypeAlias = str
"""File path on the server."""

DirPath: TypeAlias = str
"""Directory path on the server."""

Duration: TypeAlias = str
"""Duration like ``1w 2d 3h 4m 5s`` or an integer number of seconds."""

Color: TypeAlias = str
"""CSS color name."""

Regex: TypeAlias = str
"""Regular expression, as used in Python."""

FormatStr: TypeAlias = str
"""Format string as used in Python."""

DateStr: TypeAlias = str
"""ISO date string like ``2019-01-30``."""

DateTimeStr: TypeAlias = str
"""ISO datetime string like ``2019-01-30 01:02:03``."""

Url: TypeAlias = str
"""URL."""

ClassRef: TypeAlias = type | str
"""Class reference, a type, and 'ext' object or a class name."""


class Config(Data):
    """Object configuration."""

    uid: str = ''
    """Unique ID."""


class Props(Data):
    """Object properties."""

    uid: str = ''
    """Unique ID."""


class Request(Data):
    """Command request."""

    projectUid: Optional[str]
    """Unique ID of the project."""
    localeUid: Optional[str]
    """Locale ID for this request."""


class EmptyRequest(Data):
    """Empty command request."""

    pass


class ResponseError(Data):
    """Response error."""

    code: Optional[int]
    """Error code."""
    info: Optional[str]
    """Information about the error."""


class Response(Data):
    """Command response."""

    error: Optional[ResponseError]
    """Response error."""
    status: int
    """Response status or exit code."""


class ContentResponse(Response):
    """Web response with literal content."""

    asAttachment: bool
    """Serve the content as an attachment."""
    attachmentName: str
    """Name for the attachment."""
    content: bytes | str
    """Response content."""
    contentPath: str
    """Local path with the content."""
    mime: str
    """Response mime type."""
    headers: dict
    """Additional headers."""


class RedirectResponse(Response):
    """Web redirect response."""

    location: str
    """Redirect URL."""
    headers: dict
    """Additional headers."""


class AttributeType(Enum):
    """Feature attribute type."""

    bool = 'bool'
    bytes = 'bytes'
    date = 'date'
    datetime = 'datetime'
    feature = 'feature'
    featurelist = 'featurelist'
    file = 'file'
    float = 'float'
    floatlist = 'floatlist'
    geometry = 'geometry'
    int = 'int'
    intlist = 'intlist'
    str = 'str'
    strlist = 'strlist'
    time = 'time'


class GeometryType(Enum):
    """Feature geometry type.

    OGC and SQL/MM geometry types.

    References:

        OGC 06-103r4 (https://www.ogc.org/standards/sfa), https://postgis.net/docs/manual-3.3/using_postgis_dbmanagement.html
    """

    geometry = 'geometry'

    point = 'point'
    curve = 'curve'
    surface = 'surface'

    geometrycollection = 'geometrycollection'

    linestring = 'linestring'
    line = 'line'
    linearring = 'linearring'

    polygon = 'polygon'
    triangle = 'triangle'

    polyhedralsurface = 'polyhedralsurface'
    tin = 'tin'

    multipoint = 'multipoint'
    multicurve = 'multicurve'
    multilinestring = 'multilinestring'
    multipolygon = 'multipolygon'
    multisurface = 'multisurface'

    circularstring = 'circularstring'
    compoundcurve = 'compoundcurve'
    curvepolygon = 'curvepolygon'


class CliParams(Data):
    """CLI params"""
    pass
################################################################################


################################################################################
# /core/_access.pyinc


Acl: TypeAlias = list[tuple[int, str]]
"""Access Control list.

A list of tuples ``(ACL bit, role-name)`` where ``ACL bit`` is ``1`` if the access is allowed and ``0`` otherwise.
"""

AclStr: TypeAlias = str
"""A string of comma-separated pairs ``allow <role>`` or ``deny <role>``."""


class Access(Enum):
    """Access mode."""

    read = 'read'
    write = 'write'
    create = 'create'
    delete = 'delete'


class PermissionsConfig:
    """Permissions configuration."""

    all: Optional[AclStr]
    """All permissions."""
    read: Optional[AclStr]
    """Permission to read the object."""
    write: Optional[AclStr]
    """Permission to change the object."""
    create: Optional[AclStr]
    """Permission to create new objects."""
    delete: Optional[AclStr]
    """Permission to delete objects."""
    edit: Optional[AclStr]
    """A combination of write, create and delete."""


class ConfigWithAccess(Config):
    """Basic config with permissions."""

    access: Optional[AclStr]
    """Permission to read or use the object. (deprecated in 8.0)"""
    permissions: Optional[PermissionsConfig]
    """Access permissions."""
################################################################################


################################################################################
# /core/_error.pyinc


"""App Error object"""

class Error(Exception):
    """GWS error."""
    def __repr__(self):
        return log.exception_backtrace(self)[0]


class ConfigurationError(Error):
    """GWS Configuration error."""
    pass


class NotFoundError(Error):
    """Generic 'object not found' error."""
    pass


class ForbiddenError(Error):
    """Generic 'forbidden' error."""
    pass


class BadRequestError(Error):
    """Generic 'bad request' error."""
    pass


class ResponseTooLargeError(Error):
    """Generic error when a response is too large."""
    pass


##
################################################################################



################################################################################
# /spec/types.pyinc


class ApplicationManifestPlugin(Data):
    """Plugin description."""

    path: DirPath
    """Path to the plugin python module."""

    name: str = ''
    """Optional name, when omitted, the directory name will be used."""


class ApplicationManifest(Data):
    """Application manifest."""

    excludePlugins: Optional[list[str]]
    """Names of the core plugins that should be deactivated."""
    plugins: Optional[list[ApplicationManifestPlugin]]
    """Custom plugins."""
    locales: list[str]
    """Locale names supported by this application."""
    withFallbackConfig: bool = False
    """Use a minimal fallback configuration."""
    withStrictConfig: bool = False
    """Stop the application upon a configuration error."""


class ExtObjectDescriptor(Data):
    """Extension object descriptor."""

    extName: str
    """Full extension name like ``gws.ext.object.layer.wms``."""
    extType: str
    """Extension type like ``wms``."""
    classPtr: type
    """Class object."""
    ident: str
    """Identifier."""
    modName: str
    """Name of the module that contains the class."""
    modPath: str
    """Path to the module that contains the class."""


class ExtCommandDescriptor(Data):
    extName: str
    """Full extension name like ``gws.ext.object.layer.wms``."""
    extType: str
    """Extension type like ``wms``."""
    methodName: str
    """Command method name."""
    methodPtr: Callable
    """Command method."""
    request: 'Request'
    """Request sent to the command."""
    tArg: str
    """Type of the command argument."""
    tOwner: str
    """Type of the command owner."""
    owner: ExtObjectDescriptor
    """Descriptor of the command owner."""


class SpecReadOption(Enum):
    """Read options."""

    acceptExtraProps = 'acceptExtraProps'
    """Accept extra object properties."""
    allowMissing = 'allowMissing'
    """Allow otherwise required properties to be missing."""
    caseInsensitive = 'caseInsensitive'
    """Case insensitive search for properties. """
    convertValues = 'convertValues'
    """Try to convert values to specified types."""
    ignoreExtraProps = 'ignoreExtraProps'
    """Silently ignore extra object properties."""
    verboseErrors = 'verboseErrors'
    """Provide verbose error messages."""


class CommandCategory(Enum):
    """Command category."""

    api = 'api'
    """API command."""
    cli = 'cli'
    """CLI command."""
    get = 'get'
    """Web GET command."""
    post = 'post'
    """Web POST command."""


class SpecRuntime:
    """Specification runtime."""

    version: str
    """Application version."""
    manifest: ApplicationManifest
    """Application manifest."""
    appBundlePaths: list[str]
    """List of client bundle paths."""

    def read(self, value, type_name: str, path: str = '', options=Optional[set[SpecReadOption]]):
        """Read a raw value according to a spec.

         Args:
             value: Raw value from config or request.
             type_name: Object type name.
             path: Config file path.
             options: Read options.

         Returns:
             A parsed object.
         """

    def object_descriptor(self, type_name: str) -> Optional[ExtObjectDescriptor]:
        """Get an object descriptor.

         Args:
             type_name: Object type name.

         Returns:
             A descriptor or ``None`` if the type is not found.
         """

    def command_descriptor(self, command_category: CommandCategory, command_name: str) -> Optional[ExtCommandDescriptor]:
        """Get a command descriptor.

         Args:
             command_category: Command category.
             command_name: Command name.

         Returns:
             A descriptor or ``None`` if the command is not found.
         """

    def get_class(self, classref: ClassRef, ext_type: Optional[str] = None) -> Optional[type]:
        """Get a class object for a class reference.

        Args:
            classref: Class reference.
            ext_type: Extension type.

        Returns:
            A class or ``None`` if the reference is not found.
        """

    def parse_classref(self, classref: ClassRef) -> tuple[Optional[type], str, str]:
        """Parse a class reference.

        Args:
            classref: Class reference.

        Returns:
            A tuple ``(class object, class name, extension name)``.
        """
################################################################################



################################################################################
# /core/_tree.pyinc


class Object:
    """GWS object."""

    permissions: dict[Access, Acl]
    """Mapping from an access mode to a list of ACL tuples."""

    def props(self, user: 'User') -> Props:
        """Generate a ``Props`` struct for this object.

        Args:
            user: The user for which the props should be generated.
        """

    def __init__(self):
        self.permissions = {}


from .core import tree_impl

setattr(tree_impl, 'Access', Access)
setattr(tree_impl, 'Error', Error)
setattr(tree_impl, 'ConfigurationError', ConfigurationError)
setattr(tree_impl, 'Data', Data)
setattr(tree_impl, 'Props', Props)
setattr(tree_impl, 'Object', Object)

Object.__repr__ = tree_impl.object_repr


class Node(Object):
    """Configurable GWS object."""

    extName: str
    """Full extension name like ``gws.ext.object.layer.wms``."""
    extType: str
    """Extension type like ``wms``."""

    config: Config
    """Configuration for this object."""
    root: 'Root'
    """Root object."""
    parent: 'Node'
    """Parent object."""
    children: list['Node']
    """Child objects."""
    uid: str
    """Unique ID."""

    def initialize(self, config):
        return tree_impl.node_initialize(self, config)

    def pre_configure(self):
        """Pre-configuration hook."""

    def configure(self):
        """Configuration hook."""

    def post_configure(self):
        """Post-configuration hook."""

    def activate(self):
        """Activation hook."""

    def create_child(self, classref: ClassRef, config: Config = None, **kwargs):
        """Create a child object.

        Args:
            classref: Class reference.
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            A newly created object or ``None`` if the object cannot be initialized.
        """
        return tree_impl.node_create_child(self, classref, config, **kwargs)

    def create_child_if_configured(self, classref: ClassRef, config=None, **kwargs):
        """Create a child object if the configuration is not None.

        Args:
            classref: Class reference.
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            A newly created object or ``None`` if the configuration is ``None`` or the object cannot be initialized.
        """
        return tree_impl.node_create_child_if_configured(self, classref, config, **kwargs)

    def create_children(self, classref: ClassRef, configs: list[Config], **kwargs):
        """Create a list of child objects from a list of configurations.

        Args:
            classref: Class reference.
            configs: List of configurations.
            **kwargs: Additional configuration properties.

        Returns:
            A list of newly created objects.
        """
        return tree_impl.node_create_children(self, classref, configs, **kwargs)

    def cfg(self, key: str, default=None):
        """Fetch a configuration property.

        Args:
            key: Property key. If it contains dots, fetch nested properties.
            default: Default to return if the property is not found.

        Returns:
            A property value.
        """
        return tree_impl.node_cfg(self, key, default)

    def is_a(self, classref: ClassRef):
        """Check if a the node matches the class reference.

        Args:
            classref: Class reference.

        Returns:
            A boolean.
        """
        return tree_impl.is_a(self.root, self, classref)

    def find_all(self, classref: ClassRef):
        """Find all children that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            A list of objects.
        """
        return tree_impl.node_find_all(self, classref)

    def find_first(self, classref: ClassRef):
        """Find the first child that matches a specific class.

        Args:
            classref: Class reference.

        Returns:
            An object or ``None``.
        """
        return tree_impl.node_find_first(self, classref)

    def find_closest(self, classref: ClassRef):
        """Find the closest node ancestor that matches a specific class.

        Args:
            classref: Class reference.

        Returns:
            An object or ``None``.
        """

        return tree_impl.node_find_closest(self, classref)

    def find_ancestors(self, classref: ClassRef):
        """Find node ancestors that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            A list of objects.
        """
        return tree_impl.node_find_ancestors(self, classref)

    def find_descendants(self, classref: ClassRef):
        """Find node descendants that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            A list of objects in the depth-first order.
        """

        return tree_impl.node_find_descendants(self, classref)

    def enter_middleware(self, req: 'WebRequester') -> Optional['WebResponder']:
        """Begin middleware processing.

        Args:
            req: Requester object.

        Returns:
            A Responder object or ``None``.
        """

    def exit_middleware(self, req: 'WebRequester', res: 'WebResponder'):
        """Finish middleware processing.

        Args:
            req: Requester object.
            res: Current responder object.
        """

    def register_middleware(self, name: str, depends_on: Optional[list[str]] = None):
        """Register itself as a middleware handler.

        Args:
            name: Handler name.
            depends_on: List of handler names this handler depends on.
        """
        return tree_impl.node_register_middleware(self, name, depends_on)


class Root:
    """Root node of the object tree."""

    app: 'Application'
    """Application object."""
    specs: 'SpecRuntime'
    """Specs runtime."""
    configErrors: list
    """List of configuration errors."""

    nodes: list['Node']
    uidMap: dict[str, 'Node']
    uidCount: int
    configStack: list['Node']

    def __init__(self, specs: 'SpecRuntime'):
        tree_impl.root_init(self, specs)

    def initialize(self, obj, config):
        return tree_impl.root_initialize(self, obj, config)

    def post_initialize(self):
        """Post-initialization hook."""
        return tree_impl.root_post_initialize(self)

    def activate(self):
        return tree_impl.root_activate(self)

    def find_all(self, classref: ClassRef):
        """Find all objects that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            A list of objects.
        """
        return tree_impl.root_find_all(self, classref)

    def find_first(self, classref: ClassRef):
        """Find the first object that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            An object or ``None``.
        """
        return tree_impl.root_find_first(self, classref)

    def get(self, uid: str = None, classref: Optional[ClassRef] = None):
        """Get an object by its unique ID.

        Args:
            uid: Object uid.
            classref: Class reference. If provided, ensures that the object matches the reference.

        Returns:
            An object or ``None``.
        """
        return tree_impl.root_get(self, uid, classref)

    def object_count(self) -> int:
        """Return the number of objects in the tree."""
        return tree_impl.root_object_count(self)

    def create(self, classref: ClassRef, parent: Optional['Node'] = None, config: Config = None, **kwargs):
        """Create an object.

        Args:
            classref: Class reference.
            parent: Parent object.
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            A newly created object or ``None`` if the object cannot be initialized.
        """
        return tree_impl.root_create(self, classref, parent, config, **kwargs)

    def create_shared(self, classref: ClassRef, config: Config = None, **kwargs):
        """Create a shared object, attached directly to the root.

        Args:
            classref: Class reference.
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            A newly created object or ``None`` if the object cannot be initialized.
        """
        return tree_impl.root_create_shared(self, classref, config, **kwargs)

    def create_temporary(self, classref: ClassRef, config: Config = None, **kwargs):
        """Create a temporary object, not attached to the tree.

        Args:
            classref: Class reference.
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            A newly created object or ``None`` if the object cannot be initialized.
        """
        return tree_impl.root_create_temporary(self, classref, config, **kwargs)

    def create_application(self, config: Config = None, **kwargs) -> 'Application':
        """Create the Application object.

        Args:
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            The Application object.
        """
        return tree_impl.root_create_application(self, config, **kwargs)


def create_root(specs: 'SpecRuntime') -> Root:
    return Root(specs)


def props_of(obj: Object, user: 'User', *context) -> Optional['Props']:
    return tree_impl.props_of(obj, user, *context)
################################################################################



################################################################################
# /lib/image/types.pyinc


class ImageFormat(Enum):
    """Image format"""

    png8 = 'png8'
    """png 8-bit"""
    png24 = 'png24'
    """png 24-bit"""


class Image:
    """Image object."""

    def size(self) -> Size: ...

    def add_box(self, color=None) -> 'Image': ...

    def add_text(self, text: str, x=0, y=0, color=None) -> 'Image': ...

    def compose(self, other: 'Image', opacity=1) -> 'Image': ...

    def crop(self, box) -> 'Image': ...

    def paste(self, other: 'Image', where=None) -> 'Image': ...

    def resize(self, size: Size, **kwargs) -> 'Image': ...

    def rotate(self, angle: int, **kwargs) -> 'Image': ...

    def to_bytes(self, mime: Optional[str] = None) -> bytes: ...

    def to_path(self, path: str, mime: Optional[str] = None) -> str: ...

    def to_array(self) -> 'numpy.typing.NDArray': ...
################################################################################


################################################################################
# /lib/intl/types.pyinc


class Locale(Data):
    """Locale data."""

    id: str
    dateFormatLong: str
    dateFormatMedium: str
    dateFormatShort: str
    dateUnits: str
    """date unit names, e.g. 'YMD' for 'en', 'JMT' for 'de'"""
    dayNamesLong: list[str]
    dayNamesShort: list[str]
    dayNamesNarrow: list[str]
    firstWeekDay: int
    language: str
    languageName: str
    monthNamesLong: list[str]
    monthNamesShort: list[str]
    monthNamesNarrow: list[str]
    numberDecimal: str
    numberGroup: str


class DateTimeFormatType(Enum):
    """Enumeration indicating the length of the date/time format."""
    short = 'short'
    """Local short format."""
    medium = 'medium'
    """Local medium format."""
    long = 'long'
    """Local long format."""
    iso = 'iso'
    """ISO 8601 format."""


class NumberFormatType(Enum):
    """Enumeration indicating the number format."""
    decimal = 'decimal'
    """Locale decimal format."""
    grouped = 'grouped'
    """Locale grouped format."""
    currency = 'currency'
    """Locale currency format"""
    percent = 'percent'
    """Locale percent format."""


class DateFormatter:
    """Used for date formatting"""

    def format(self, fmt: DateTimeFormatType | str, date=None) -> str:
        """Formats the date with respect to the locale.

        Args:
            fmt: Format type or a `strftime` format string
            date: Date, if none is given the current date will be used as default.

        Returns:
            A formatted date string.
        """


class TimeFormatter:
    """Used for date formatting"""

    def format(self, fmt: DateTimeFormatType | str, time=None) -> str:
        """Formats the time with respect to the locale.

        Args:
            fmt: Format type or a `strftime` format string
            time: Date, if none is given the current time will be used as default.

        Returns:
            A formatted time string.
        """


class NumberFormatter:
    """Used for number formatting"""

    def format(self, fmt: NumberFormatType | str, n, *args, **kwargs) -> str:
        """Formats the number with respect to the locale.

        Args:
            fmt: Format type or a python `format` string
            n: Number.
            kwargs: Passes the currency parameter forward.

        Returns:
            A formatted number.
        """
################################################################################


################################################################################
# /lib/job/types.pyinc


class JobState(Enum):
    """Background job state."""

    init = 'init'
    """The job is being created."""
    open = 'open'
    """The job is just created and waiting for start."""
    running = 'running'
    """The job is running."""
    complete = 'complete'
    """The job has been completed successfully."""
    error = 'error'
    """There was an error."""
    cancel = 'cancel'
    """The job was cancelled."""


class Job:
    """Background Job object."""

    error: str
    payload: dict
    state: JobState
    uid: str
    user: 'User'

    def run(self): ...

    def update(self, payload: Optional[dict] = None, state: Optional[JobState] = None, error: Optional[str] = None): ...

    def cancel(self): ...

    def remove(self): ...
################################################################################


################################################################################
# /lib/metadata/types.pyinc


class MetadataLink(Data):
    """Link metadata."""

    description: Optional[str]
    format: Optional[str]
    formatVersion: Optional[str]
    function: Optional[str]
    mimeType: Optional[str]
    scheme: Optional[str]
    title: Optional[str]
    type: Optional[str]
    url: Optional[Url]


class MetadataAccessConstraint(Data):
    """Metadata AccessConstraint."""

    title: Optional[str]
    type: Optional[str]


class MetadataLicense(Data):
    """Metadata License."""

    title: Optional[str]
    url: Optional[Url]


class MetadataAttribution(Data):
    """Metadata Attribution."""

    title: Optional[str]
    url: Optional[Url]


class Metadata(Data):
    """Metadata."""

    abstract: Optional[str]
    accessConstraints: Optional[list[MetadataAccessConstraint]]
    attribution: Optional[MetadataAttribution]
    authorityIdentifier: Optional[str]
    authorityName: Optional[str]
    authorityUrl: Optional[str]
    catalogCitationUid: Optional[str]
    catalogUid: Optional[str]
    fees: Optional[str]
    image: Optional[str]
    keywords: Optional[list[str]]
    language3: Optional[str]
    language: Optional[str]
    languageName: Optional[str]
    license: Optional[MetadataLicense]
    name: Optional[str]
    parentIdentifier: Optional[str]
    title: Optional[str]

    contactAddress: Optional[str]
    contactAddressType: Optional[str]
    contactArea: Optional[str]
    contactCity: Optional[str]
    contactCountry: Optional[str]
    contactEmail: Optional[str]
    contactFax: Optional[str]
    contactOrganization: Optional[str]
    contactPerson: Optional[str]
    contactPhone: Optional[str]
    contactPosition: Optional[str]
    contactProviderName: Optional[str]
    contactProviderSite: Optional[str]
    contactRole: Optional[str]
    contactUrl: Optional[str]
    contactZip: Optional[str]

    dateBegin: Optional[str]
    dateCreated: Optional[str]
    dateEnd: Optional[str]
    dateUpdated: Optional[str]

    inspireKeywords: Optional[list[str]]
    inspireMandatoryKeyword: Optional[str]
    inspireDegreeOfConformity: Optional[str]
    inspireResourceType: Optional[str]
    inspireSpatialDataServiceType: Optional[str]
    inspireSpatialScope: Optional[str]
    inspireSpatialScopeName: Optional[str]
    inspireTheme: Optional[str]
    inspireThemeName: Optional[str]
    inspireThemeNameEn: Optional[str]

    isoMaintenanceFrequencyCode: Optional[str]
    isoQualityConformanceExplanation: Optional[str]
    isoQualityConformanceQualityPass: Optional[bool]
    isoQualityConformanceSpecificationDate: Optional[str]
    isoQualityConformanceSpecificationTitle: Optional[str]
    isoQualityLineageSource: Optional[str]
    isoQualityLineageSourceScale: Optional[int]
    isoQualityLineageStatement: Optional[str]
    isoRestrictionCode: Optional[str]
    isoServiceFunction: Optional[str]
    isoScope: Optional[str]
    isoScopeName: Optional[str]
    isoSpatialRepresentationType: Optional[str]
    isoTopicCategories: Optional[list[str]]
    isoSpatialResolution: Optional[str]

    metaLinks: Optional[list[MetadataLink]]
    serviceMetaLink: Optional[MetadataLink]
    extraLinks: Optional[list[MetadataLink]]
################################################################################


################################################################################
# /lib/style/types.pyinc


class StyleValues(Data):
    """CSS Style values."""

    fill: Color

    stroke: Color
    stroke_dasharray: list[int]
    stroke_dashoffset: int
    stroke_linecap: Literal['butt', 'round', 'square']
    stroke_linejoin: Literal['bevel', 'round', 'miter']
    stroke_miterlimit: int
    stroke_width: int

    marker: Literal['circle', 'square', 'arrow', 'cross']
    marker_fill: Color
    marker_size: int
    marker_stroke: Color
    marker_stroke_dasharray: list[int]
    marker_stroke_dashoffset: int
    marker_stroke_linecap: Literal['butt', 'round', 'square']
    marker_stroke_linejoin: Literal['bevel', 'round', 'miter']
    marker_stroke_miterlimit: int
    marker_stroke_width: int

    with_geometry: Literal['all', 'none']
    with_label: Literal['all', 'none']

    label_align: Literal['left', 'right', 'center']
    label_background: Color
    label_fill: Color
    label_font_family: str
    label_font_size: int
    label_font_style: Literal['normal', 'italic']
    label_font_weight: Literal['normal', 'bold']
    label_line_height: int
    label_max_scale: int
    label_min_scale: int
    label_offset_x: int
    label_offset_y: int
    label_padding: list[int]
    label_placement: Literal['start', 'end', 'middle']
    label_stroke: Color
    label_stroke_dasharray: list[int]
    label_stroke_dashoffset: int
    label_stroke_linecap: Literal['butt', 'round', 'square']
    label_stroke_linejoin: Literal['bevel', 'round', 'miter']
    label_stroke_miterlimit: int
    label_stroke_width: int

    point_size: int
    icon: str

    offset_x: int
    offset_y: int


class StyleProps(Props):
    """CSS Style properties."""

    cssSelector: Optional[str]
    text: Optional[str]
    values: Optional[dict]


class Style:
    """CSS Style object."""

    cssSelector: str
    text: str
    values: StyleValues
################################################################################


################################################################################
# /lib/xmlx/types.pyinc


class XmlNamespace(Data):
    """XML namespace."""

    uid: str
    """Unique ID."""
    xmlns: str
    """Default prefix for this Namespace."""
    uri: Url
    """Namespace uri."""
    schemaLocation: Url
    """Namespace schema location."""
    version: str
    """Namespace version."""
    extendsGml: bool
    """Namespace schema extends the GML3 schema."""


class XmlElement(Iterable):
    """XML Element.


    Extends ``ElementTree.Element`` (https://docs.python.org/3/library/xml.etree.elementtree.html#element-objects).
    """

    tag: str
    """Tag name, with an optional namespace in the Clark notation."""

    text: Optional[str]
    """Text before first subelement."""

    tail: Optional[str]
    """Text after this element's end tag."""

    attrib: dict
    """Dictionary of element attributes."""

    name: str
    """Element name (tag without a namespace)."""

    lcName: str
    """Element name (tag without a namespace) in lower case."""

    caseInsensitive: bool
    """Element is case-insensitive."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator['XmlElement']: ...

    def __getitem__(self, item: int) -> 'XmlElement': ...

    def append(self, subelement: 'XmlElement'):
        """Adds the element subelement to the end of this element’s internal list of subelements."""

    def attr(self, key: str, default=None):
        """Finds the value for a given key in the ``XmlElementImpl``.

        Args:
            key: Key of the attribute.
            default: The default return.

        Returns:
            The vale of the key, If the key is not found the default is returned.
        """

    def clear(self):
        """Resets an element."""

    def extend(self, subelements: Iterable['XmlElement']):
        """Appends subelements from a sequence object with zero or more elements."""

    def find(self, path: str) -> Optional['XmlElement']:
        """Finds first matching element by tag name or path."""

    def findall(self, path: str) -> list['XmlElement']:
        """Finds all matching subelements by name or path."""

    def findtext(self, path: str, default: Optional[str] = None) -> str:
        """Finds text for first matching element by name or path."""

    def get(self, key: str, default=None):
        """Returns the value to a given key."""

    def insert(self, index: int, subelement: 'XmlElement'):
        """Inserts subelement at the given position in this element."""

    def items(self) -> Iterable[tuple[str, Any]]:
        """Returns the element attributes as a sequence of (name, value) pairs."""

    def iter(self, tag: Optional[str] = None) -> Iterable['XmlElement']:
        """Creates a tree iterator."""

    def iterfind(self, path: Optional[str] = None) -> Iterable['XmlElement']:
        """Returns an iterable of all matching subelements by name or path."""

    def itertext(self) -> Iterable[str]:
        """Creates a text iterator and returns all inner text."""

    def keys(self) -> Iterable[str]:
        """Returns the elements attribute names as a list."""

    def remove(self, other: 'XmlElement'):
        """Removes the other element from the element."""

    def set(self, key: str, value: Any):
        """Set the attribute key on the element to value."""

    # extensions

    def add(self, tag: str, attrib: Optional[dict] = None, **extra) -> 'XmlElement':
        """Creates a new ``XmlElementImpl`` and adds it as a child.

        Args:
            tag: XML tag.
            attrib: XML attributes ``{key, value}``.

        Returns:
            A XmlElementImpl.
        """

    def children(self) -> list['XmlElement']:
        """Returns the children of the current ``XmlElementImpl``."""

    def findfirst(self, *paths) -> Optional['XmlElement']:
        """Returns the first element in the current element.

        Args:
            paths: Path as ``tag/tag2/tag3`` to the Element to search in.

        Returns:
            Returns the first found element.
        """

    def textof(self, *paths) -> str:
        """Returns the text of a given child-element.

        Args:
            paths: Path as ``tag/tag2/tag3`` to the Element.

        Returns:
            The text of the element.
        """

    def textlist(self, *paths, deep=False) -> list[str]:
        """Collects texts from child-elements.

        Args:
            paths: Path as ``tag/tag2/tag3`` to the Element to collect texts from.
            deep: If ``False`` it only looks into direct children, otherwise it searches for texts in the complete children-tree.

        Returns:
            A list containing all the text from the child-elements.
        """

    def textdict(self, *paths, deep=False) -> dict[str, str]:
        """Collects texts from child-elements.

        Args:
            paths: Path as ``tag/tag2/tag3`` to the Element to collect texts from.
            deep: If ``False`` it only looks into direct children, otherwise it searches for texts in the complete children-tree.

        Returns:
            A dict containing all the text from the child-elements.
        """

    def to_string(
            self,
            extra_namespaces: Optional[list[XmlNamespace]] = None,
            compact_whitespace: bool = False,
            remove_namespaces: bool = False,
            with_namespace_declarations: bool = False,
            with_schema_locations: bool = False,
            with_xml_declaration: bool = False,
    ) -> str:
        """Converts the Element object to a string.

        Args:
            extra_namespaces: Extra namespaces to add to the document.
            compact_whitespace: Remove all whitespace outside of tags and elements.
            remove_namespaces: Remove all namespace references.
            with_namespace_declarations: Include the namespace declarations.
            with_schema_locations: Include schema locations.
            with_xml_declaration: Include the xml declaration.

        Returns:
            An XML string.
        """

    def to_dict(self) -> dict:
        """Creates a dictionary from an XmlElement object."""
################################################################################


################################################################################
# /lib/uom/types.pyinc


class Uom(Enum):
    """Unit of measure."""

    mi = 'mi'
    """statute mile (EPSG 9093)"""
    us_ch = 'us-ch'
    """us survey chain (EPSG 9033)"""
    us_ft = 'us-ft'
    """us survey foot (EPSG 9003)"""
    us_in = 'us-in'
    """us survey inch us_in"""
    us_mi = 'us-mi'
    """us survey mile (EPSG 9035)"""
    us_yd = 'us-yd'
    """us survey yard us_yd"""
    cm = 'cm'
    """centimetre (EPSG 1033)"""
    ch = 'ch'
    """chain (EPSG 9097)"""
    dm = 'dm'
    """decimeter dm"""
    deg = 'deg'
    """degree (EPSG 9102)"""
    fath = 'fath'
    """fathom (EPSG 9014)"""
    ft = 'ft'
    """foot (EPSG 9002)"""
    grad = 'grad'
    """grad (EPSG 9105)"""
    inch = 'in'
    """inch in"""
    km = 'km'
    """kilometre (EPSG 9036)"""
    link = 'link'
    """link (EPSG 9098)"""
    m = 'm'
    """metre (EPSG 9001)"""
    mm = 'mm'
    """millimetre (EPSG 1025)"""
    kmi = 'kmi'
    """nautical mile (EPSG 9030)"""
    rad = 'rad'
    """radian (EPSG 9101)"""
    yd = 'yd'
    """yard (EPSG 9096)"""
    px = 'px'
    """pixel"""
    pt = 'pt'
    """point"""


UomValue: TypeAlias = tuple[float, Uom]
"""A value with a unit."""

UomValueStr: TypeAlias = str
"""A value with a unit like ``5mm``."""

UomPoint: TypeAlias = tuple[float, float, Uom]
"""A Point with a unit."""

UomPointStr: TypeAlias = list[str]
"""A Point with a unit like ``["1mm", "2mm"]``."""

UomSize: TypeAlias = tuple[float, float, Uom]
"""A Size with a unit."""

UomSizeStr: TypeAlias = list[str]
"""A Size with a unit like ``["1mm", "2mm"]``."""

UomExtent: TypeAlias = tuple[float, float, float, float, Uom]
"""Extent with a unit."""

UomExtentStr: TypeAlias = list[str]
"""Extent with a unit like ``["1mm", "2mm", "3mm", "4mm"]``."""
################################################################################



################################################################################
# /gis/crs/types.pyinc


CrsName: TypeAlias = int | str
"""A CRS code like ``EPSG:3857`` or a SRID like ``3857``."""


class CrsFormat(Enum):
    """CRS name format."""

    none = ''
    crs = 'crs'
    """Like ``crs84``."""
    srid = 'srid'
    """Like ``3857``."""
    epsg = 'epsg'
    """Like ``EPSG:3857``."""
    url = 'url'
    """Like ``http://www.opengis.net/gml/srs/epsg.xml#3857``."""
    uri = 'uri'
    """Like ``http://www.opengis.net/def/crs/epsg/0/3857``."""
    urnx = 'urnx'
    """Like ``urn:x-ogc:def:crs:EPSG:3857``."""
    urn = 'urn'
    """Like ``urn:ogc:def:crs:EPSG::3857``."""


class Axis(Enum):
    """Axis orientation."""

    xy = 'xy'
    yx = 'yx'


class Bounds(Data):
    """Geo-referenced extent."""

    crs: 'Crs'
    extent: Extent


class Crs:
    """Coordinate reference system."""

    srid: int
    """CRS SRID."""
    axis: Axis
    """Axis orientation."""
    uom: Uom
    """CRS unit."""
    isGeographic: bool
    """This CRS is geographic."""
    isProjected: bool
    """This CRS is projected."""
    isYX: bool
    """This CRS has a lat/lon axis."""
    proj4text: str
    """Proj4 definition."""
    wkt: str
    """WKT definition."""

    epsg: str
    """Name in the "epsg" format."""
    urn: str
    """Name in the "urn" format."""
    urnx: str
    """Name in the "urnx" format."""
    url: str
    """Name in the "url" format."""
    uri: str
    """Name in the "uri" format."""

    name: str
    """CRS name."""
    base: int
    """Base CRS code."""
    datum: str
    """Datum."""

    wgsExtent: Extent
    """CRS Extent in the WGS projection."""
    extent: Extent
    """CRS own Extent."""

    def axis_for_format(self, fmt: 'CrsFormat') -> Axis:
        """Get the axis depending on the string format.

        We adhere to the GeoServer convention here:
        https://docs.geoserver.org/latest/en/user/services/wfs/axis_order.html
        """

    def transform_extent(self, extent: Extent, crs_to: 'Crs') -> Extent:
        """Transform an Extent from this CRS to another.

        Args:
            extent: Extent.
            crs_to: Target CRS.

        Returns:
            A transformed Extent.
        """

    def transformer(self, crs_to: 'Crs') -> Callable:
        """Create a transformer function to another CRS.

        Args:
            crs_to: Target CRS.

        Returns:
            A function.
        """

    def to_string(self, fmt: Optional['CrsFormat'] = None) -> str:
        """Return a string representation of the CRS.

        Args:
            fmt: Format to use.

        Returns:
            A string.
        """

    def to_geojson(self) -> dict:
        """Return a geojson representation of the CRS (as per GJ2008).

        Returns:
            A GeoJson dict.

        References:
            https://geojson.org/geojson-spec#named-crs
        """
################################################################################


################################################################################
# /gis/render/types.pyinc


class MapView(Data):
    """Map view."""

    bounds: Bounds
    center: Point
    rotation: int
    scale: int
    mmSize: Size
    pxSize: Size
    dpi: int


class MapRenderInputPlaneType(Enum):
    """Map render input plane type."""

    features = 'features'
    image = 'image'
    imageLayer = 'imageLayer'
    svgLayer = 'svgLayer'
    svgSoup = 'svgSoup'


class MapRenderInputPlane(Data):
    """Map render input plane."""

    type: MapRenderInputPlaneType
    features: list['Feature']
    image: 'Image'
    layer: 'Layer'
    opacity: float
    soupPoints: list[Point]
    soupTags: list[Any]
    styles: list['Style']
    subLayers: list[str]


class MapRenderInput(Data):
    """Map render input."""

    backgroundColor: int
    bbox: Extent
    center: Point
    crs: 'Crs'
    dpi: int
    mapSize: UomSize
    notify: Callable
    planes: list['MapRenderInputPlane']
    project: 'Project'
    rotation: int
    scale: int
    user: 'User'
    visibleLayers: Optional[list['Layer']]


class MapRenderOutputPlaneType(Enum):
    """Map render output plane type."""

    image = 'image'
    path = 'path'
    svg = 'svg'


class MapRenderOutputPlane(Data):
    """Map render output plane."""

    type: MapRenderOutputPlaneType
    path: str
    elements: list[XmlElement]
    image: 'Image'


class MapRenderOutput(Data):
    """Map render output."""

    planes: list['MapRenderOutputPlane']
    view: MapView


class LayerRenderInputType(Enum):
    """Layer render input type."""

    box = 'box'
    xyz = 'xyz'
    svg = 'svg'


class LayerRenderInput(Data):
    """Layer render input."""

    boxBuffer: int
    boxSize: int
    extraParams: dict
    project: 'Project'
    style: 'Style'
    type: LayerRenderInputType
    user: 'User'
    view: MapView
    x: int
    y: int
    z: int


class LayerRenderOutput(Data):
    """Layer render output."""

    content: bytes
    tags: list[XmlElement]
################################################################################


################################################################################
# /gis/source/types.pyinc


class TileMatrix(Data):
    """WMTS TileMatrix object."""

    uid: str
    scale: float
    x: float
    y: float
    width: float
    height: float
    tileWidth: float
    tileHeight: float
    extent: Extent


class TileMatrixSet(Data):
    """WMTS TileMatrixSet object."""

    uid: str
    crs: 'Crs'
    matrices: list[TileMatrix]


class SourceStyle(Data):
    """Generic OGC Style."""

    isDefault: bool
    legendUrl: Url
    metadata: 'Metadata'
    name: str


class SourceLayer(Data):
    """Generic OGC Layer."""

    aLevel: int
    aPath: str
    aUid: str

    dataSource: dict
    metadata: 'Metadata'

    supportedCrs: list['Crs']
    wgsBounds: Bounds

    isExpanded: bool
    isGroup: bool
    isImage: bool
    isQueryable: bool
    isVisible: bool

    layers: list['SourceLayer']

    name: str
    title: str

    legendUrl: Url
    opacity: int
    scaleRange: list[float]

    styles: list[SourceStyle]
    defaultStyle: Optional[SourceStyle]

    tileMatrixIds: list[str]
    tileMatrixSets: list[TileMatrixSet]
    imageFormat: str
    resourceUrls: dict

    sourceId: str
    properties: dict
################################################################################



################################################################################
# /server/types.pyinc


class ServerManager(Node):
    """Server configuration manager."""

    templates: list['Template']

    def create_server_configs(self, target_dir: str, script_path: str, pid_paths: dict):
        """Create server configuration files."""


class ServerMonitor(Node):
    """File Monitor facility."""

    def add_directory(self, path: str, pattern: 'Regex'):
        """Add a directory to monitor.

        Args:
            path: Directory path.
            pattern: Regex pattern for files to watch.
        """

    def add_file(self, path: str):
        """Add a file to watch.

        Args:
            path: File path.
        """

    def start(self):
        """Start the monitor."""
################################################################################



################################################################################
# /base/feature/types.pyinc


FeatureUid: TypeAlias = str
"""Unique Feature id."""

class FeatureRecord(Data):
    """Raw data from a feature source."""

    attributes: dict
    meta: dict
    uid: Optional[str]
    shape: Optional['Shape']


class FeatureProps(Props):
    """Feature Proprieties."""

    attributes: dict
    cssSelector: str
    errors: Optional[list['ModelValidationError']]
    createWithFeatures: Optional[list['FeatureProps']]
    isNew: bool
    modelUid: str
    uid: str
    views: dict


class Feature:
    """Feature object."""

    attributes: dict
    category: str
    cssSelector: str
    errors: list['ModelValidationError']
    isNew: bool
    model: 'Model'
    props: 'FeatureProps'
    record: 'FeatureRecord'
    views: dict
    createWithFeatures: list['Feature']
    insertedPrimaryKey: str

    def get(self, name: str, default=None) -> Any: ...

    def has(self, name: str) -> bool: ...

    def set(self, name: str, value: Any) -> 'Feature': ...

    def raw(self, name: str) -> Any: ...

    def render_views(self, templates: list['Template'], **kwargs) -> 'Feature': ...

    def shape(self) -> Optional['Shape']: ...

    def to_geojson(self, user: 'User') -> dict: ...

    def to_svg(self, view: 'MapView', label: Optional[str] = None, style: Optional['Style'] = None) -> list[XmlElement]: ...

    def transform_to(self, crs: 'Crs') -> 'Feature': ...

    def uid(self) -> FeatureUid: ...
################################################################################


################################################################################
# /base/shape/types.pyinc


class ShapeProps(Props):
    """Shape properties."""

    crs: str
    geometry: dict


class Shape(Object):
    """Geo-referenced geometry."""

    type: GeometryType
    """Geometry type."""

    crs: 'Crs'
    """CRS of this shape."""

    x: Optional[float]
    """X-coordinate for Point geometries, None otherwise."""

    y: Optional[float]
    """Y-coordinate for Point geometries, None otherwise."""

    # common props

    def area(self) -> float:
        """Computes the area of the geometry."""

    def bounds(self) -> Bounds:
        """Returns a Bounds object that bounds this shape."""

    def centroid(self) -> 'Shape':
        """Returns a centroid as a Point shape."""

    # formats

    def to_wkb(self) -> bytes:
        """Returns a WKB representation of this shape as a binary string."""

    def to_wkb_hex(self) -> str:
        """Returns a WKB representation of this shape as a hex string."""

    def to_ewkb(self) -> bytes:
        """Returns an EWKB representation of this shape as a binary string."""

    def to_ewkb_hex(self) -> str:
        """Returns an EWKB representation of this shape as a hex string."""

    def to_wkt(self) -> str:
        """Returns a WKT representation of this shape."""

    def to_ewkt(self) -> str:
        """Returns an EWKT representation of this shape."""

    def to_geojson(self, always_xy=False) -> dict:
        """Returns a GeoJSON representation of this shape."""

    def to_props(self) -> ShapeProps:
        """Returns a GeoJSON representation of this shape."""

    # predicates (https://shapely.readthedocs.io/en/stable/manual.html#predicates-and-relationships)

    def is_empty(self) -> bool:
        """Returns True if this shape is empty."""

    def is_ring(self) -> bool:
        """Returns True if this shape is a ring."""

    def is_simple(self) -> bool:
        """Returns True if this shape is 'simple'."""

    def is_valid(self) -> bool:
        """Returns True if this shape is valid."""

    def equals(self, other: 'Shape') -> bool:
        """Returns True if this shape is equal to the other."""

    def contains(self, other: 'Shape') -> bool:
        """Returns True if this shape contains the other."""

    def covers(self, other: 'Shape') -> bool:
        """Returns True if this shape covers the other."""

    def covered_by(self, other: 'Shape') -> bool:
        """Returns True if this shape is covered by the other."""

    def crosses(self, other: 'Shape') -> bool:
        """Returns True if this shape crosses the other."""

    def disjoint(self, other: 'Shape') -> bool:
        """Returns True if this shape does not intersect with the other."""

    def intersects(self, other: 'Shape') -> bool:
        """Returns True if this shape intersects with the other."""

    def overlaps(self, other: 'Shape') -> bool:
        """Returns True if this shape overlaps the other."""

    def touches(self, other: 'Shape') -> bool:
        """Returns True if this shape touches the other."""

    def within(self, other: 'Shape') -> bool:
        """Returns True if this shape is within the other."""

    # set operations

    def union(self, others: list['Shape']) -> 'Shape':
        """Computes a union of this shape and other shapes."""

    def intersection(self, *others: 'Shape') -> 'Shape':
        """Computes an intersection of this shape and other shapes."""

    # convertors

    def to_multi(self) -> 'Shape':
        """Converts a singly-geometry shape to a multi-geometry one."""

    def to_type(self, new_type: 'GeometryType') -> 'Shape':
        """Converts a geometry to another type."""

    # misc

    def tolerance_polygon(self, tolerance, quad_segs=None) -> 'Shape':
        """Builds a buffer polygon around the shape."""

    def transformed_to(self, crs: 'Crs') -> 'Shape':
        """Returns this shape transformed to another CRS."""
################################################################################



################################################################################
# /base/action/types.pyinc


class ActionManager(Node):
    """Action manager."""

    def actions_for_project(self, project: 'Project', user: 'User') -> list['Action']:
        """Get a list of actions for a Project, to which a User has access to."""

    def find_action(self, project: Optional['Project'], ext_type: str, user: 'User') -> Optional['Action']:
        """Locate an Action object.

        Args:
            project: Project to se
            ext_type:
            user:

        Returns:

        """

    def prepare_action(
            self,
            command_category: CommandCategory,
            command_name: str,
            params: dict,
            user: 'User',
            read_options=None,
    ) -> tuple[Callable, Request]: ...


class Action(Node):
    pass
################################################################################


################################################################################
# /base/auth/types.pyinc


class User(Object):
    """User object."""

    authProvider: 'AuthProvider'
    """User authorization provider."""

    attributes: dict
    """Custom user attributes."""
    authToken: str
    """Token used for authorization."""
    displayName: str
    """User display name."""
    isGuest: bool
    """User is a Guest."""
    localUid: str
    """User uid within its authorization provider."""
    loginName: str
    """User login name."""
    roles: set[str]
    """User roles."""
    uid: str
    """Global user uid."""

    def acl_bit(self, access: 'Access', obj: 'Object') -> Optional[int]:
        """Get the ACL bit for a specific object.

        Args:
            access: Access mode.
            obj: Requested object.

        Returns:
            ``1`` or ``0`` if the user's permissions have the bit and ``None`` otherwise.
        """

    def can(self, access: Access, obj: 'Object', *context) -> bool:
        """Check if the user can access an object.

        Args:
            access: Access mode.
            obj: Requested object.
            *context: Further objects to check.

        Returns:
            ``True`` is access is granted.
        """

    def can_create(self, obj: 'Object', *context) -> bool:
        """Check if the user has "create" permission on an object."""

    def can_delete(self, obj: 'Object', *context) -> bool:
        """Check if the user has "delete" permission on an object."""

    def can_read(self, obj: 'Object', *context) -> bool:
        """Check if the user has "read" permission on an object."""

    def can_use(self, obj: 'Object', *context) -> bool:
        """Check if the user has "read" permission on an object."""

    def can_write(self, obj: 'Object', *context) -> bool:
        """Check if the user has "write" permission on an object."""

    def can_edit(self, obj: 'Object', *context) -> bool:
        """Check if the user has "edit" permissions on an object."""

    def acquire(self, uid: str = None, classref: Optional[ClassRef] = None, access: Optional[Access] = None) -> Optional['Object']:
        """Get a readable object by uid.

        Args:
            uid: Object uid.
            classref: Class reference. If provided, ensures that the object matches the reference.
            access: Access mode, assumed ``Access.read`` if omitted.

        Returns:
            A readable object or ``None`` if the object does not exists or user doesn't have a permission.
        """

    def require(self, uid: str = None, classref: Optional[ClassRef] = None, access: Optional[Access] = None) -> 'Object':
        """Get a readable object by uid and fail if not found.

        Args:
            uid: Object uid.
            classref: Class reference. If provided, ensures that the object matches the reference.
            access: Access mode, assumed ``Access.read`` if omitted.

        Returns:
            A readable object.

        Raises:
            ``NotFoundError`` if the object doesn't exist.
            ``ForbiddenError`` if the user cannot read the object.
        """

    def require_project(self, uid: str = None) -> 'Project':
        """Get a readable Project object.

        Args:
            uid: Project uid.

        Returns:
            A Project object.
        """

    def require_layer(self, uid = None) -> 'Layer':
        """Get a readable Layer object.

        Args:
            uid: Layer uid.

        Returns:
            A Layer object.
        """


class AuthManager(Node):
    """Authentication manager."""

    guestSession: 'AuthSession'
    """Preconfigured Guest session."""

    guestUser: 'User'
    """Preconfigured Guest user."""
    systemUser: 'User'
    """Preconfigured System user."""

    providers: list['AuthProvider']
    """Authentication providers."""
    methods: list['AuthMethod']
    """Authentication methods."""
    mfa: list['AuthMfa']
    """Authentication MFA handlers."""

    sessionMgr: 'AuthSessionManager'
    """Session manager."""

    def authenticate(self, method: 'AuthMethod', credentials: Data) -> Optional['User']:
        """Authenticate a user.

        Args:
            method: Authentication method.
            credentials: Credentials object.

        Returns:
            An authenticated User or ``None`` if authentication failed.
        """

    def get_user(self, user_uid: str) -> Optional['User']:
        """Get a User by its global uid.

        Args:
            user_uid: Global user uid.
        Returns:
            A User or ``None``.
        """

    def get_provider(self, uid: str) -> Optional['AuthProvider']:
        """Get an authentication Provider by its uid.

        Args:
            uid: Uid.
        Returns:
            A Provider or ``None``.
        """

    def get_method(self, uid: str) -> Optional['AuthMethod']:
        """Get an authentication Method by its uid.

        Args:
            uid: Uid.
        Returns:
            A Method or ``None``.
        """

    def get_mfa(self, uid: str) -> Optional['AuthMfa']:
        """Get an authentication Provider by its uid.

        Args:
            uid: Uid.
        Returns:
            A Provider or ``None``.
        """

    def serialize_user(self, user: 'User') -> str:
        """Return a string representation of a User.

        Args:
            user: A User object.

        Returns:
            A json string.
        """

    def unserialize_user(self, ser: str) -> Optional['User']:
        """Restore a User object from a serialized representation.

        Args:
            ser: A json string.

        Returns:
            A User object.
        """


class AuthMethod(Node):
    """Authentication Method."""

    authMgr: 'AuthManager'


    secure: bool
    """Method is only allowed in a secure context."""

    def open_session(self, req: 'WebRequester') -> Optional['AuthSession']:
        """Attempt to open a Session for a Requester.

        Args:
            req: Requester object.

        Returns:
            A Session or ``None``.
        """

    def close_session(self, req: 'WebRequester', res: 'WebResponder') -> bool:
        """Close a previously opened Session.

        Args:
            req: Requester object.
            res: Responder object.

        Returns:
            True if the Session was successfully closed.
        """


class AuthMfa(Node):
    """Authentication MFA handler."""

    autoStart: bool
    lifeTime: int
    maxAttempts: int
    maxRestarts: int

    def start(self, user: 'User'): ...

    def is_valid(self, user: 'User') -> bool: ...

    def cancel(self, user: 'User'): ...

    def verify(self, user: 'User', request: Data) -> bool: ...

    def restart(self, user: 'User') -> bool: ...


class AuthProvider(Node):
    """Authentication Provider."""

    allowedMethods: list[str]
    """List of Method types allowed to be used with this Provider."""

    def get_user(self, local_uid: str) -> Optional['User']:
        """Get a User from its local uid.

        Args:
            local_uid: User local uid.

        Returns:
            A User or ``None``.
        """

    def authenticate(self, method: 'AuthMethod', credentials: Data) -> Optional['User']:
        """Authenticate a user.

        Args:
            method: Authentication method.
            credentials: Credentials object.

        Returns:
            An authenticated User or ``None`` if authentication failed.
        """

    def serialize_user(self, user: 'User') -> str:
        """Return a string representation of a User.

        Args:
            user: A User object.

        Returns:
            A json string.
        """

    def unserialize_user(self, ser: str) -> Optional['User']:
        """Restore a User object from a serialized representation.

        Args:
            ser: A json string.

        Returns:
            A User object.
        """


class AuthSession:
    """Authentication session."""

    uid: str
    """Session uid."""
    method: Optional['AuthMethod']
    """Authentication method that created the session."""
    user: 'User'
    """Authorized User."""
    data: dict
    """Session data."""
    created: 'datetime.datetime'
    """Session create time."""
    updated: 'datetime.datetime'
    """Session update time."""
    isChanged: bool
    """Session has changed since the last update.."""

    def get(self, key: str, default=None):
        """Get a session data value.

        Args:
            key: Value name.
            default: Default value.

        Returns:
            A value or the default.
        """

    def set(self, key: str, value):
        """Set a session data value.

        Args:
            key: Value name.
            value: A value.
        """


class AuthSessionManager(Node):
    """Authentication session Manager."""

    lifeTime: int
    """Session lifetime in seconds."""

    def create(self, method: 'AuthMethod', user: 'User', data: Optional[dict] = None) -> 'AuthSession':
        """Create a new Session,

        Args:
            method: Auth Method that creates the Session.
            user: 'User' for which the Session is created.
            data: Session data.

        Returns:
            A new Session.
        """

    def delete(self, sess: 'AuthSession'):
        """Delete a Session.

        Args:
            sess: Session object.
        """

    def delete_all(self):
        """Delete all Sessions.
        """

    def get(self, uid: str) -> Optional['AuthSession']:
        """Get Session by its uid.

        Args:
            uid: Session uid.

        Returns:
            A Session or ``None``.
        """

    def get_valid(self, uid: str) -> Optional['AuthSession']:
        """Get a valid Session by its uid.

        Args:
            uid: Session uid.

        Returns:
            A Session or ``None`` if uid does not exists or the Session is not valid.
        """

    def get_all(self) -> list['AuthSession']:
        """Get all sessions."""

    def save(self, sess: 'AuthSession'):
        """Save the Session state into a persistent storage.

        Args:
            sess: Session object.
        """

    def touch(self, sess: 'AuthSession'):
        """Update the Session last activity timestamp.

        Args:
            sess: Session object.
        """

    def cleanup(self):
        """Remove invalid Sessions from the storage.
        """
################################################################################



################################################################################
# /base/layer/types.pyinc


class LayerDisplayMode(Enum):
    """Layer display mode."""

    box = 'box'
    """Display a layer as one big image (WMS-alike)."""
    tile = 'tile'
    """Display a layer in a tile grid."""
    client = 'client'
    """Draw a layer in the client."""


class LayerClientOptions(Data):
    """Client options for a layer."""

    expanded: bool
    """A layer is expanded in the list view."""
    unlisted: bool
    """A layer is hidden in the list view."""
    selected: bool
    """A layer is initially selected."""
    hidden: bool
    """A layer is initially hidden."""
    unfolded: bool
    """A layer is not listed, but its children are."""
    exclusive: bool
    """Only one of this layer's children is visible at a time."""


class TileGrid(Data):
    """Tile grid."""

    uid: str
    bounds: Bounds
    origin: Origin
    resolutions: list[float]
    tileSize: int


class LayerCache(Data):
    """Layer cache."""

    maxAge: int
    maxLevel: int
    requestBuffer: int
    requestTiles: int


class FeatureLoadingStrategy(Enum):
    """Loading strategy for features."""

    all = 'all'
    """Load all features."""
    bbox = 'bbox'
    """Load only features in the current map extent."""
    lazy = 'lazy'
    """Load features on demand."""


class LayerOwsOptions(Data):
    """Layer options for OWS services."""

    layerName: str
    featureName: str
    xmlNamespace: 'XmlNamespace'
    geometryName: str


class Layer(Node):
    """Layer object."""

    canRenderBox: bool
    canRenderSvg: bool
    canRenderXyz: bool

    isEnabledForOws: bool
    isGroup: bool
    isSearchable: bool

    hasLegend: bool

    bounds: Bounds
    wgsExtent: Extent
    mapCrs: 'Crs'
    clientOptions: LayerClientOptions
    displayMode: LayerDisplayMode
    loadingStrategy: FeatureLoadingStrategy
    imageFormat: str
    opacity: float
    resolutions: list[float]
    title: str

    owsOptions: 'LayerOwsOptions'

    grid: Optional[TileGrid]
    cache: Optional[LayerCache]

    metadata: 'Metadata'
    legend: Optional['Legend']
    legendUrl: str

    finders: list['Finder']
    templates: list['Template']
    models: list['Model']

    layers: list['Layer']

    sourceLayers: list['SourceLayer']

    def render(self, lri: LayerRenderInput) -> Optional['LayerRenderOutput']: ...

    def get_features_for_view(self, search: 'SearchQuery', user: 'User', view_names: Optional[list[str]] = None) -> list['Feature']: ...

    def render_legend(self, args: Optional[dict] = None) -> Optional['LegendRenderOutput']: ...

    def url_path(self, kind: str) -> str: ...
################################################################################


################################################################################
# /base/legend/types.pyinc


class LegendRenderOutput(Data):
    """Legend render output."""

    html: str
    image: 'Image'
    image_path: str
    size: Size
    mime: str


class Legend(Node):
    """Legend object."""

    def render(self, args: Optional[dict] = None) -> Optional[LegendRenderOutput]: ...
################################################################################


################################################################################
# /base/map/types.pyinc


class Map(Node):
    """Map object."""

    rootLayer: 'Layer'

    bounds: Bounds
    center: Point
    coordinatePrecision: int
    initResolution: float
    resolutions: list[float]
    title: str
    wgsExtent: Extent
################################################################################



################################################################################
# /base/model/types.pyinc


class ModelValidationError(Data):
    """Validation error."""

    fieldName: str
    message: str


class ModelOperation(Enum):
    """Model operation."""

    read = 'read'
    create = 'create'
    update = 'update'
    delete = 'delete'


class ModelReadMode(Enum):
    """Model reading mode."""

    render = 'render'
    search = 'search'
    list = 'list'
    form = 'form'


class ModelDbSelect(Data):
    """Database select statement."""

    columns: list['sqlalchemy.Column']
    geometryWhere: list
    keywordWhere: list
    where: list
    order: list


class ModelContext(Data):
    """Model context."""

    op: ModelOperation
    readMode: ModelReadMode
    user: 'User'
    project: 'Project'
    relDepth: int = 0
    maxDepth: int = 0
    search: 'SearchQuery'
    dbSelect: ModelDbSelect
    dbConnection: 'sqlalchemy.Connection'


EmptyValue = object()
"""Special value for empty fields."""

ErrorValue = object()
"""Special value for invalid fields."""


class ModelWidget(Node):
    """Model widget."""

    supportsTableView: bool = True


class ModelValidator(Node):
    """Model Validator."""

    message: str
    ops: set[ModelOperation]

    def validate(self, field: 'ModelField', feature: 'Feature', mc: ModelContext) -> bool: ...


class ModelValue(Node):
    """Model value."""

    isDefault: bool
    ops: set[ModelOperation]

    def compute(self, field: 'ModelField', feature: 'Feature', mc: 'ModelContext'): ...


class ModelField(Node):
    """Model field."""

    name: str
    title: str

    attributeType: AttributeType

    widget: Optional['ModelWidget'] = None

    values: list['ModelValue']
    validators: list['ModelValidator']

    isPrimaryKey: bool
    isRequired: bool
    isUnique: bool
    isAuto: bool

    supportsFilterSearch: bool = False
    supportsGeometrySearch: bool = False
    supportsKeywordSearch: bool = False

    model: 'Model'

    def before_select(self, mc: ModelContext): ...

    def after_select(self, features: list['Feature'], mc: ModelContext): ...

    def before_create(self, feature: 'Feature', mc: ModelContext): ...

    def after_create(self, feature: 'Feature', mc: ModelContext): ...

    def before_create_related(self, to_feature: 'Feature', mc: ModelContext): ...

    def after_create_related(self, to_feature: 'Feature', mc: ModelContext): ...

    def before_update(self, feature: 'Feature', mc: ModelContext): ...

    def after_update(self, feature: 'Feature', mc: ModelContext): ...

    def before_delete(self, feature: 'Feature', mc: ModelContext): ...

    def after_delete(self, feature: 'Feature', mc: ModelContext): ...

    def do_init(self, feature: 'Feature', mc: ModelContext): ...

    def do_init_related(self, to_feature: 'Feature', mc: ModelContext): ...

    def do_validate(self, feature: 'Feature', mc: ModelContext): ...

    def from_props(self, feature: 'Feature', mc: ModelContext): ...

    def to_props(self, feature: 'Feature', mc: ModelContext): ...

    def from_record(self, feature: 'Feature', mc: ModelContext): ...

    def to_record(self, feature: 'Feature', mc: ModelContext): ...

    def related_models(self) -> list['Model']: ...

    def find_relatable_features(self, search: 'SearchQuery', mc: ModelContext) -> list['Feature']: ...

    def raw_to_python(self, feature: 'Feature', value, mc: ModelContext): ...

    def prop_to_python(self, feature: 'Feature', value, mc: ModelContext): ...

    def python_to_raw(self, feature: 'Feature', value, mc: ModelContext): ...

    def python_to_prop(self, feature: 'Feature', value, mc: ModelContext): ...

    def describe(self) -> Optional['ColumnDescription']: ...


class Model(Node):
    """Data Model."""

    defaultSort: list['SearchSort']
    fields: list['ModelField']
    geometryCrs: Optional['Crs']
    geometryName: str
    geometryType: Optional[GeometryType]
    isEditable: bool
    loadingStrategy: 'FeatureLoadingStrategy'
    title: str
    uidName: str
    withTableView: bool

    def find_features(self, search: 'SearchQuery', mc: ModelContext) -> list['Feature']: ...

    def get_features(self, uids: Iterable[str | int], mc: ModelContext) -> list['Feature']: ...

    def init_feature(self, feature: 'Feature', mc: ModelContext): ...

    def create_feature(self, feature: 'Feature', mc: ModelContext) -> FeatureUid: ...

    def update_feature(self, feature: 'Feature', mc: ModelContext) -> FeatureUid: ...

    def delete_feature(self, feature: 'Feature', mc: ModelContext) -> FeatureUid: ...

    def validate_feature(self, feature: 'Feature', mc: ModelContext) -> bool: ...

    def feature_from_props(self, props: 'FeatureProps', mc: ModelContext) -> 'Feature': ...

    def feature_to_props(self, feature: 'Feature', mc: ModelContext) -> 'FeatureProps': ...

    def feature_to_view_props(self, feature: 'Feature', mc: ModelContext) -> 'FeatureProps': ...

    def describe(self) -> Optional['DataSetDescription']: ...

    def field(self, name: str) -> Optional['ModelField']: ...

    def related_models(self) -> list['Model']: ...


class ModelManager(Node):
    """Model manager."""

    def get_model(self, uid: str, user: 'User' = None, access: Access = None) -> Optional['Model']: ...

    def find_model(self, *objects, user: 'User' = None, access: Access = None) -> Optional['Model']: ...

    def editable_models(self, project: 'Project', user: 'User') -> list['Model']: ...

    def default_model(self) -> 'Model': ...
################################################################################


################################################################################
# /base/database/types.pyinc


class DatabaseModel(Model):
    """Database-based data model."""

    db: 'DatabaseProvider'
    sqlFilter: str
    tableName: str

    def table(self) -> 'sqlalchemy.Table': ...

    def column(self, column_name: str) -> 'sqlalchemy.Column': ...

    def uid_column(self) -> 'sqlalchemy.Column': ...




class ColumnDescription(Data):
    """Database column description."""

    columnIndex: int
    comment: str
    default: str
    geometrySrid: int
    geometryType: GeometryType
    isAutoincrement: bool
    isNullable: bool
    isPrimaryKey: bool
    isUnique: bool
    hasDefault: bool
    name: str
    nativeType: str
    options: dict
    type: AttributeType


class RelationshipDescription(Data):
    """Database relationship description."""

    name: str
    schema: str
    fullName: str
    foreignKeys: str
    referredKeys: str


class DataSetDescription(Data):
    """Description of a database Table or a GDAL Dataset."""

    columns: list[ColumnDescription]
    columnMap: dict[str, ColumnDescription]
    fullName: str
    geometryName: str
    geometrySrid: int
    geometryType: GeometryType
    name: str
    schema: str


class DatabaseManager(Node):
    """Database manager."""

    providers: list['DatabaseProvider']

    def create_provider(self, cfg: Config, **kwargs) -> 'DatabaseProvider': ...

    def find_provider(self, uid: Optional[str] = None, ext_type: Optional[str] = None) -> Optional['DatabaseProvider']: ...


DatabaseTableAlike: TypeAlias = Union['sqlalchemy.Table', str]
"""SA ``Table`` object or a string table name."""


class DatabaseProvider(Node):
    """Database Provider.

    A database Provider wraps SQLAlchemy ``Engine`` and ``Connection`` objects
    and provides common db functionality.
    """

    url: str
    """Connection url."""

    def column(self, table: DatabaseTableAlike, column_name: str) -> 'sqlalchemy.Column':
        """SA ``Column`` object for a specific column."""

    def connect(self) -> ContextManager['sqlalchemy.Connection']:
        """Context manager for a SA ``Connection``.

        Context calls to this method can be nested. An inner call is a no-op, as no new connection is created.
        Only the outermost connection is closed upon exit::

            with db.connect():
                ...
                with db.connect(): # no-op
                    ...
                # connection remains open
                ...
            # connection closed
        """

    def describe(self, table: DatabaseTableAlike) -> 'DataSetDescription':
        """Describe a table."""

    def count(self, table: DatabaseTableAlike) -> int:
        """Return table record count or 0 if the table does not exist."""

    def engine(self, **kwargs) -> 'sqlalchemy.Engine':
        """SA ``Engine`` object for this provider."""

    def has_column(self, table: DatabaseTableAlike, column_name: str) -> bool:
        """Check if a specific column exists."""

    def has_table(self, table_name: str) -> bool:
        """Check if a specific table exists."""

    def join_table_name(self, schema: str, name: str) -> str:
        """Create a full table name from the schema and table names."""

    def split_table_name(self, table_name: str) -> tuple[str, str]:
        """Split a full table name into the schema and table names."""

    def table(self, table_name: str, **kwargs) -> 'sqlalchemy.Table':
        """SA ``Table`` object for a specific table."""

    def table_bounds(self, table: DatabaseTableAlike) -> Optional[Bounds]:
        """Compute a bounding box for the table primary geometry."""
################################################################################



################################################################################
# /base/ows/types.pyinc


class OwsProtocol(Enum):
    """Supported OWS protocol."""

    WMS = 'WMS'
    WMTS = 'WMTS'
    WCS = 'WCS'
    WFS = 'WFS'
    CSW = 'CSW'


class OwsAuthorization(Data):
    type: str
    username: str
    password: str


class OwsVerb(Enum):
    """OWS verb."""

    CreateStoredQuery = 'CreateStoredQuery'
    DescribeCoverage = 'DescribeCoverage'
    DescribeFeatureType = 'DescribeFeatureType'
    DescribeLayer = 'DescribeLayer'
    DescribeRecord = 'DescribeRecord'
    DescribeStoredQueries = 'DescribeStoredQueries'
    DropStoredQuery = 'DropStoredQuery'
    GetCapabilities = 'GetCapabilities'
    GetFeature = 'GetFeature'
    GetFeatureInfo = 'GetFeatureInfo'
    GetFeatureWithLock = 'GetFeatureWithLock'
    GetLegendGraphic = 'GetLegendGraphic'
    GetMap = 'GetMap'
    GetPrint = 'GetPrint'
    GetPropertyValue = 'GetPropertyValue'
    GetRecordById = 'GetRecordById'
    GetRecords = 'GetRecords'
    GetTile = 'GetTile'
    ListStoredQueries = 'ListStoredQueries'
    LockFeature = 'LockFeature'
    Transaction = 'Transaction'


class OwsOperation(Data):
    """OWS operation."""

    allowedParameters: dict[str, list[str]]
    constraints: dict[str, list[str]]
    formats: list[str]
    params: dict[str, str]
    postUrl: Url
    preferredFormat: str
    url: Url
    verb: OwsVerb


class OwsCapabilities(Data):
    """OWS capabilities structure."""

    metadata: 'Metadata'
    operations: list['OwsOperation']
    sourceLayers: list['SourceLayer']
    tileMatrixSets: list['TileMatrixSet']
    version: str


class OwsService(Node):
    """OWS Service."""

    isRasterService: bool
    """Service provides raster services."""
    isVectorService: bool
    """Service provides vector services."""

    alwaysXY: bool
    """Force lon/lat order for geographic projections."""
    metadata: 'Metadata'
    """Service metadata."""
    name: str
    """Service name."""
    protocol: OwsProtocol
    """Supported protocol."""
    supportedBounds: list[Bounds]
    """Supported bounds."""
    supportedVersions: list[str]
    """Supported versions."""
    supportedOperations: list['OwsOperation']
    """Supported operations."""
    templates: list['Template']
    """Service templates."""
    updateSequence: str
    """Service update sequence."""
    withInspireMeta: bool
    """Include INSPIRE metadata."""
    withStrictParams: bool
    """Strict parameter checking."""

    def handle_request(self, req: 'WebRequester') -> ContentResponse:
        """Handle a service request."""


class OwsProvider(Node):
    """OWS services Provider."""

    alwaysXY: bool
    authorization: Optional['OwsAuthorization']
    bounds: Optional[Bounds]
    forceCrs: 'Crs'
    maxRequests: int
    metadata: 'Metadata'
    operations: list['OwsOperation']
    protocol: 'OwsProtocol'
    sourceLayers: list['SourceLayer']
    url: Url
    version: str
    wgsExtent: Optional[Extent]

    def get_operation(self, verb: 'OwsVerb', method: Optional['RequestMethod'] = None) -> Optional['OwsOperation']: ...

    def get_features(self, args: 'SearchQuery', source_layers: list['SourceLayer']) -> list['FeatureRecord']: ...
################################################################################


################################################################################
# /base/printer/types.pyinc


class PrintPlaneType(Enum):
    """Print plane type."""

    bitmap = 'bitmap'
    url = 'url'
    features = 'features'
    raster = 'raster'
    vector = 'vector'
    soup = 'soup'


class PrintPlane(Data):
    """Print plane."""

    type: PrintPlaneType

    opacity: Optional[float]
    cssSelector: Optional[str]

    bitmapData: Optional[bytes]
    bitmapMode: Optional[str]
    bitmapWidth: Optional[int]
    bitmapHeight: Optional[int]

    url: Optional[str]

    features: Optional[list['FeatureProps']]

    layerUid: Optional[str]
    subLayers: Optional[list[str]]

    soupPoints: Optional[list[Point]]
    soupTags: Optional[list[Any]]


class PrintMap(Data):
    """Map properties for printing."""

    backgroundColor: Optional[int]
    bbox: Optional[Extent]
    center: Optional[Point]
    planes: list[PrintPlane]
    rotation: Optional[int]
    scale: int
    styles: Optional[list['StyleProps']]
    visibleLayers: Optional[list[str]]


class PrintRequestType(Enum):
    """Type of the print request."""

    template = 'template'
    map = 'map'


class PrintRequest(Request):
    """Print request."""

    type: PrintRequestType

    args: Optional[dict]
    crs: Optional[CrsName]
    outputFormat: Optional[str]
    maps: Optional[list[PrintMap]]

    printerUid: Optional[str]
    dpi: Optional[int]
    outputSize: Optional[Size]


class PrintJobResponse(Response):
    """Print job information response."""

    jobUid: str
    progress: int
    state: 'JobState'
    stepType: str
    stepName: str
    url: str


class Printer(Node):
    """Printer object."""

    title: str
    template: 'Template'
    models: list['Model']
    qualityLevels: list['TemplateQualityLevel']


class PrinterManager(Node):
    """Print Manager."""

    def printers_for_project(self, project: 'Project', user: 'User') -> list['Printer']: ...

    def start_job(self, request: PrintRequest, user: 'User') -> 'Job': ...

    def get_job(self, uid: str, user: 'User') -> Optional['Job']: ...

    def run_job(self, request: PrintRequest, user: 'User'): ...

    def cancel_job(self, job: 'Job'): ...

    def result_path(self, job: 'Job') -> str: ...

    def status(self, job: 'Job') -> PrintJobResponse: ...
################################################################################


################################################################################
# /base/project/types.pyinc


class Client(Node):
    """GWS Client control object."""

    options: dict
    elements: list


class Project(Node):
    """Project object."""

    assetsRoot: Optional['WebDocumentRoot']
    client: 'Client'

    localeUids: list[str]
    map: 'Map'
    metadata: 'Metadata'

    actions: list['Action']
    finders: list['Finder']
    models: list['Model']
    printers: list['Printer']
    templates: list['Template']
    owsServices: list['OwsService']
################################################################################


################################################################################
# /base/search/types.pyinc


class SearchSort(Data):
    """Search sort specification."""

    fieldName: str
    reverse: bool


class SearchOgcFilter(Data):
    """Search filter."""

    name: str
    operator: str
    shape: 'Shape'
    subFilters: list['SearchOgcFilter']
    value: str


class SearchQuery(Data):
    """Search query."""

    access: Access
    all: bool
    bounds: Bounds
    extraColumns: list
    extraParams: dict
    extraWhere: list
    keyword: str
    layers: list['Layer']
    limit: int
    ogcFilter: SearchOgcFilter
    project: 'Project'
    relDepth: int
    resolution: float
    shape: 'Shape'
    sort: list[SearchSort]
    tolerance: 'UomValue'
    uids: list[str]


class SearchResult(Data):
    """Search result."""

    feature: 'Feature'
    layer: 'Layer'
    finder: 'Finder'


class TextSearchType(Enum):
    """Text search type."""

    exact = 'exact'
    """Match the whole string."""
    begin = 'begin'
    """Match the beginning of the string."""
    end = 'end'
    """Match the end of the string."""
    any = 'any'
    """Match any substring."""
    like = 'like'
    """Use the percent sign as a placeholder."""


class TextSearchOptions(Data):
    """Text search options."""

    type: TextSearchType
    """Type of the search."""
    minLength: int = 0
    """Minimal pattern length."""
    caseSensitive: bool = False
    """Use the case sensitive search."""


class SortOptions(Data):
    """Sort options."""
    fieldName: str
    reverse: bool = False


class SearchManager(Node):
    """Search Manager."""

    def run_search(self, search: 'SearchQuery', user: 'User') -> list['SearchResult']: ...


class Finder(Node):
    """Finder object."""

    title: str

    supportsFilterSearch: bool = False
    supportsGeometrySearch: bool = False
    supportsKeywordSearch: bool = False

    withFilter: bool
    withGeometry: bool
    withKeyword: bool

    templates: list['Template']
    models: list['Model']
    sourceLayers: list['SourceLayer']

    tolerance: 'UomValue'

    def run(self, search: SearchQuery, user: 'User', layer: Optional['Layer'] = None) -> list['Feature']: ...

    def can_run(self, search: SearchQuery, user: 'User') -> bool: ...
################################################################################


################################################################################
# /base/storage/types.pyinc


class StorageManager(Node):
    """Storage manager."""

    providers: list['StorageProvider']

    def create_provider(self, cfg: Config, **kwargs) -> 'StorageProvider': ...

    def find_provider(self, uid: Optional[str] = None) -> Optional['StorageProvider']: ...



class StorageRecord(Data):
    """Storage record."""

    name: str
    userUid: str
    data: str
    created: int
    updated: int


class StorageProvider(Node):
    """Storage provider."""

    def list_names(self, category: str) -> list[str]: ...

    def read(self, category: str, name: str) -> Optional['StorageRecord']: ...

    def write(self, category: str, name: str, data: str, user_uid: str): ...

    def delete(self, category: str, name: str): ...
################################################################################


################################################################################
# /base/template/types.pyinc


class TemplateArgs(Data):
    """Template arguments."""

    app: 'Application'
    """Application object."""
    gwsVersion: str
    """GWS version."""
    gwsBaseUrl: str
    """GWS server base url."""
    locale: 'Locale'
    """Current locale."""
    date: 'DateFormatter'
    """Locale-dependent date formatter."""
    time: 'TimeFormatter'
    """Locale-dependent time formatter."""
    number: 'NumberFormatter'
    """Locale-dependent number formatter."""


class TemplateRenderInput(Data):
    """Template render input."""

    args: dict | Data
    crs: 'Crs'
    dpi: int
    localeUid: str
    maps: list[MapRenderInput]
    mimeOut: str
    notify: Callable
    project: 'Project'
    user: 'User'


class TemplateQualityLevel(Data):
    """Template quality level."""

    name: str
    dpi: int


class Template(Node):
    """Template object."""

    mapSize: UomSize
    """Default map size for the template."""
    mimeTypes: list[str]
    """MIME types the template can generate."""
    pageSize: UomSize
    """Default page size for printing."""
    pageMargin: UomExtent
    """Default page margin for printing."""
    subject: str
    """Template subject (category)."""
    title: str
    """Template title."""

    def render(self, tri: TemplateRenderInput) -> Response:
        """Render the template and return the generated response."""


class TemplateManager(Node):
    """Template manager."""

    def find_template(self, *objects, user: 'User' = None, subject: str = None, mime: str = None) -> Optional['Template']: ...

    def template_from_path(self, path: str) -> Optional['Template']: ...
################################################################################


################################################################################
# /base/web/types.pyinc


class RequestMethod(Enum):
    """Web request method."""

    GET = 'GET'
    HEAD = 'HEAD'
    POST = 'POST'
    PUT = 'PUT'
    DELETE = 'DELETE'
    CONNECT = 'CONNECT'
    OPTIONS = 'OPTIONS'
    TRACE = 'TRACE'
    PATCH = 'PATCH'


class WebRequester:
    """Web Requester object."""

    environ: dict
    """Request environment."""
    method: RequestMethod
    """Request method."""
    root: 'Root'
    """Object tree root."""
    site: 'WebSite'
    """Website the request is processed for."""
    params: dict
    """GET parameters."""
    command: str
    """Command name to execute."""

    session: 'AuthSession'
    """Current session."""
    user: 'User'
    """Current use."""

    isApi: bool
    """The request is an 'api' request."""
    isGet: bool
    """The request is a 'get' request."""
    isPost: bool
    """The request is a 'post' request."""
    isSecure: bool
    """The request is secure."""

    def initialize(self):
        """Initialize the Requester."""

    def cookie(self, key: str, default: str = '') -> str:
        """Get a cookie.

        Args:
            key: Cookie name.
            default: Default value.

        Returns:
            A cookie value.
        """

    def header(self, key: str, default: str = '') -> str:
        """Get a header.

        Args:
            key: Header name.
            default: Default value.

        Returns:
            A header value.
        """

    def param(self, key: str, default: str = '') -> str:
        """Get a GET parameter.

        Args:
            key: Parameter name.
            default: Default value.

        Returns:
            A parameter value.
        """

    def env(self, key: str, default: str = '') -> str:
        """Get an environment variable.

        Args:
            key: Variable name.
            default: Default value.

        Returns:
            A variable value.
        """

    def data(self) -> Optional[bytes]:
        """Get POST data.

        Returns:
            Data bytes or ``None`` if request is not a POST.
        """

    def text(self) -> Optional[str]:
        """Get POST data as a text.

        Returns:
            Data string or ``None`` if request is not a POST.
        """

    def content_responder(self, response: ContentResponse) -> 'WebResponder':
        """Return a Responder object for a content response.

        Args:
            response: Response object.

        Returns:
            A Responder.
        """

    def redirect_responder(self, response: RedirectResponse) -> 'WebResponder':
        """Return a Responder object for a redirect response.

        Args:
            response: Response object.

        Returns:
            A Responder.
        """

    def api_responder(self, response: Response) -> 'WebResponder':
        """Return a Responder object for an Api (structured) response.

        Args:
            response: Response object.

        Returns:
            A Responder.
        """

    def error_responder(self, exc: Exception) -> 'WebResponder':
        """Return a Responder object for an Exception.

        Args:
            exc: An Exception.

        Returns:
            A Responder.
        """

    def url_for(self, request_path: str, **kwargs) -> str:
        """Return a canonical Url for the given request path.

        Args:
            request_path: Request path.
            **kwargs: Additional GET parameters.

        Returns:
            An URL.
        """

    def set_session(self, session: 'AuthSession'):
        """Attach a session to the requester.

        Args:
            session: A Session object.
        """


class WebResponder:
    """Web Responder object."""

    status: int
    """Response status."""

    def send_response(self, environ: dict, start_response: Callable):
        """Send the response to the client.

        Args:
            environ: WSGI environment.
            start_response: WSGI ``start_response`` function.
        """

    def set_cookie(self, key: str, value: str, **kwargs):
        """Set a cookie.

        Args:
            key: Cookie name.
            value: Cookie value.
            **kwargs: Cookie options.
        """

    def delete_cookie(self, key: str, **kwargs):
        """Delete a cookie.

        Args:
            key: Cookie name.
            **kwargs: Cookie options.
        """

    def set_status(self, status: int):
        """Set the response status.

        Args:
            status: HTTP status code.
        """

    def add_header(self, key: str, value: str):
        """Add a header.

        Args:
            key: Header name.
            value: Header value.
        """


class WebDocumentRoot(Data):
    """Web document root."""

    dir: DirPath
    """Local directory."""
    allowMime: list[str]
    """Allowed mime types."""
    denyMime: list[str]
    """Restricted mime types."""


class WebRewriteRule(Data):
    """Rewrite rule."""

    pattern: Regex
    """URL matching pattern."""
    target: str
    """Rule target, with dollar placeholders."""
    options: dict
    """Extra options."""
    reversed: bool
    """Reversed rewrite rule."""


class WebCors(Data):
    """CORS options."""

    allowCredentials: bool
    allowHeaders: str
    allowMethods: str
    allowOrigin: str


class WebManager(Node):
    """Web manager."""

    sites: list['WebSite']
    """Configured web sites."""

    def site_from_environ(self, environ: dict) -> 'WebSite':
        """Returns a site object for the given request environment.

        Args:
            environ: WSGI environment.

        Returns:
            A Site object.
        """


class WebSite(Node):
    """Web site."""

    assetsRoot: Optional[WebDocumentRoot]
    """Root directory for assets."""
    corsOptions: WebCors
    """CORS options."""
    errorPage: Optional['Template']
    """Error page template."""
    host: str
    """Host name for this site."""
    rewriteRules: list[WebRewriteRule]
    """Rewrite rule."""
    staticRoot: WebDocumentRoot
    """Root directory for static files."""

    def url_for(self, req: 'WebRequester', path: str, **kwargs) -> str:
        """Rewrite a request path to an Url.

        Args:
            req: Web Requester.
            path: Raw request path.
            **kwargs: Extra GET params.

        Returns:
            A rewritten URL.
        """
################################################################################


################################################################################
# /base/xml/types.pyinc


class XmlManager(Node):
    """XML namespaces and options manager."""

    def add_namespace(self, cfg: Config) -> XmlNamespace:
        """Create and register a custom namespace."""
################################################################################



################################################################################
# /base/application/types.pyinc


class MiddlewareManager(Node):
    def register(self, obj: Node, name: str, depends_on: Optional[list[str]] = None):
        """Register an object as a middleware."""

    def objects(self) -> list[Node]:
        """Return a list of registered middleware objects."""


class Application(Node):
    """The main Application object."""

    client: 'Client'
    localeUids: list[str]
    metadata: 'Metadata'
    monitor: 'ServerMonitor'
    version: str
    versionString: str
    defaultPrinter: 'Printer'

    actionMgr: 'ActionManager'
    authMgr: 'AuthManager'
    databaseMgr: 'DatabaseManager'
    modelMgr: 'ModelManager'
    printerMgr: 'PrinterManager'
    searchMgr: 'SearchManager'
    storageMgr: 'StorageManager'
    templateMgr: 'TemplateManager'
    serverMgr: 'ServerManager'
    webMgr: 'WebManager'
    middlewareMgr: 'MiddlewareManager'
    xmlMgr: 'XmlManager'

    actions: list['Action']
    projects: list['Project']
    finders: list['Finder']
    templates: list['Template']
    printers: list['Printer']
    models: list['Model']
    owsServices: list['OwsService']

    def project(self, uid: str) -> Optional['Project']:
        """Get a Project object by its uid."""

    def helper(self, ext_type: str) -> Optional['Node']:
        """Get a Helper object by its extension type."""

    def developer_option(self, key: str):
        """Get a value of a developer option."""
################################################################################
