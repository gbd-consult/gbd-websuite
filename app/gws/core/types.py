"""GWS types.

Definitions in this module are star-imported in the ``gws`` package and are available
as ``gws.<type>``::

    import gws

    some_var: gws.Bounds = ...

There are several kinds of objects defined here:

- Type aliases
- Enumerations
- Plain data objects, extending the ``Data`` object
- Interfaces for rich objects (extend ``IObject``)
- Interfaces for configurable objects (extend ``INode``)

"""

from .data import Data

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
    Protocol
)

from gws.types import Enum

if TYPE_CHECKING:
    import datetime
    import sqlalchemy
    import sqlalchemy.orm
    import numpy.typing

# mypy: disable-error-code="empty-body"

# ----------------------------------------------------------------------------------------------------------------------
# common type aliases and enums

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


Measurement: TypeAlias = tuple[float, Uom]
"""A value with a unit like ``5mm``."""

MPoint: TypeAlias = tuple[float, float, Uom]
"""Point with a unit."""

MSize: TypeAlias = tuple[float, float, Uom]
"""Size with a unit like ``["1mm", "2mm"]``."""

MExtent: TypeAlias = tuple[float, float, float, float, Uom]
"""Extent with a unit like ``["1mm", "2mm", "3mm", "4mm"]``."""

FilePath: TypeAlias = str
"""File path on the server."""

DirPath: TypeAlias = str
"""Directory path on the server."""

Duration: TypeAlias = int
"""Duration like ``1w 2d 3h 4m 5s`` or an integer number of seconds."""

Color: TypeAlias = str
"""CSS color name."""

Regex: TypeAlias = str
"""Regular expression, as used in Python."""

FormatStr: TypeAlias = str
"""Format string as used in Python."""

Date: TypeAlias = str
"""ISO date string like ``2019-01-30``."""

DateTime: TypeAlias = str
"""ISO datetime string like ``2019-01-30 01:02:03``."""

Url: TypeAlias = str
"""URL."""


# ----------------------------------------------------------------------------------------------------------------------
# application manifest


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


# ----------------------------------------------------------------------------------------------------------------------
# basic objects

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


# ----------------------------------------------------------------------------------------------------------------------
# spec runtime


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


class ISpecRuntime(Protocol):
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


# ----------------------------------------------------------------------------------------------------------------------
# permissions


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

    read: Optional[AclStr]
    """Permission to read the object."""
    write: Optional[AclStr]
    """Permission to change the object."""
    create: Optional[AclStr]
    """Permission to create new objects."""
    delete: Optional[AclStr]
    """Permission to delete objects."""
    edit: Optional[AclStr]
    """A combination of read, write, create and delete."""


class ConfigWithAccess(Config):
    """Basic config with permissions."""

    access: Optional[AclStr]
    """Permission to read or use the object."""
    permissions: Optional[PermissionsConfig]
    """Additional permissions."""


# ----------------------------------------------------------------------------------------------------------------------
# foundation interfaces

class IObject(Protocol):
    """GWS object."""

    permissions: dict[Access, Acl]
    """Mapping from an access mode to a list of ACL tuples."""

    def props(self, user: 'IUser') -> Props:
        """Generate a ``Props`` struct for this object.

        Args:
            user: The user for which the props should be generated.
        """


class INode(IObject, Protocol):
    """Configurable GWS object."""

    extName: str
    """Full extension name like ``gws.ext.object.layer.wms``."""
    extType: str
    """Extension type like ``wms``."""

    config: Config
    """Configuration for this object."""
    root: 'IRoot'
    """Root object."""
    parent: 'INode'
    """Parent object."""
    children: list['INode']
    """Child objects."""
    uid: str
    """Unique ID."""

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

    def create_child_if_configured(self, classref: ClassRef, config=None, **kwargs):
        """Create a child object if the configuration is not None.

        Args:
            classref: Class reference.
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            A newly created object or ``None`` if the configuration is ``None`` or the object cannot be initialized.
        """

    def create_children(self, classref: ClassRef, configs: list[Config], **kwargs):
        """Create a list of child objects from a list of configurations.

        Args:
            classref: Class reference.
            configs: List of configurations.
            **kwargs: Additional configuration properties.

        Returns:
            A list of newly created objects.
        """

    def cfg(self, key: str, default=None):
        """Fetch a configuration property.

        Args:
            key: Property key. If it contains dots, fetch nested properties.
            default: Default to return if the property is not found.

        Returns:
            A property value.
        """

    def find_all(self, classref: ClassRef):
        """Find all children that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            A list of objects.
        """

    def find_first(self, classref: ClassRef):
        """Find the first child that matches a specific class.

        Args:
            classref: Class reference.

        Returns:
            An object or ``None``.
        """

    def find_closest(self, classref: ClassRef):
        """Find the closest node ancestor that matches a specific class.

        Args:
            classref: Class reference.

        Returns:
            An object or ``None``.
        """

    def find_ancestors(self, classref: ClassRef):
        """Find node ancestors that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            A list of objects.
        """

    def find_descendants(self, classref: ClassRef):
        """Find node descendants that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            A list of objects in the depth-first order.
        """

    def enter_middleware(self, req: 'IWebRequester') -> Optional['IWebResponder']:
        """Begin middleware processing.

        Args:
            req: Requester object.

        Returns:
            A Responder object or ``None``.
        """

    def exit_middleware(self, req: 'IWebRequester', res: 'IWebResponder'):
        """Finish middleware processing.

        Args:
            req: Requester object.
            res: Current responder object.
        """

    def register_middleware(self, name: str, depends_on=Optional[list[str]]):
        """Register itself as a middleware handler.

        Args:
            name: Handler name.
            depends_on: List of handler names this handler depends on.
        """


class IRoot(Protocol):
    """Root node of the object tree."""

    app: 'IApplication'
    """Application object."""
    specs: 'ISpecRuntime'
    """Specs runtime."""
    configErrors: list
    """List of configuration errors."""

    def post_initialize(self):
        """Post-initialization hook."""

    def activate(self):
        """Activation hook."""

    def find_all(self, classref: ClassRef):
        """Find all objects that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            A list of objects.
        """

    def find_first(self, classref: ClassRef):
        """Find the first object that match a specific class.

        Args:
            classref: Class reference.

        Returns:
            An object or ``None``.
        """

    def get(self, uid: str, classref: Optional[ClassRef] = None):
        """Get an object by its unique ID.

        Args:
            uid: Object uid.
            classref: Class reference. If provided, ensures that the object matches the reference.

        Returns:
            An object or ``None``.
        """

    def object_count(self) -> int:
        """Return the number of objects in the tree."""

    def create(self, classref: ClassRef, parent: Optional['INode'] = None, config: Config = None, **kwargs):
        """Create an object.

        Args:
            classref: Class reference.
            parent: Parent object.
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            A newly created object or ``None`` if the object cannot be initialized.
        """

    def create_shared(self, classref: ClassRef, config: Config = None, **kwargs):
        """Create a shared object, attached directly to the root.

        Args:
            classref: Class reference.
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            A newly created object or ``None`` if the object cannot be initialized.
        """

    def create_temporary(self, classref: ClassRef, config: Config = None, **kwargs):
        """Create a temporary object, not attached to the tree.

        Args:
            classref: Class reference.
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            A newly created object or ``None`` if the object cannot be initialized.
        """

    def create_application(self, config: Config = None, **kwargs) -> 'IApplication':
        """Create the Application object.

        Args:
            config: Configuration.
            **kwargs: Additional configuration properties.

        Returns:
            The Application object.
        """


class IProvider(INode, Protocol):
    """Provider object."""

    pass


# ----------------------------------------------------------------------------------------------------------------------
# requests and responses


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


class IWebRequester(Protocol):
    """Web Requester object."""

    environ: dict
    """Request environment."""
    method: RequestMethod
    """Request method."""
    root: 'IRoot'
    """Object tree root."""
    site: 'IWebSite'
    """Website the request is processed for."""
    params: dict
    """GET parameters."""
    command: str
    """Command name to execute."""

    session: 'IAuthSession'
    """Current session."""
    user: 'IUser'
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

    def content_responder(self, response: ContentResponse) -> 'IWebResponder':
        """Return a Responder object for a content response.

        Args:
            response: Response object.

        Returns:
            A Responder.
        """

    def redirect_responder(self, response: RedirectResponse) -> 'IWebResponder':
        """Return a Responder object for a redirect response.

        Args:
            response: Response object.

        Returns:
            A Responder.
        """

    def api_responder(self, response: Response) -> 'IWebResponder':
        """Return a Responder object for an Api (structured) response.

        Args:
            response: Response object.

        Returns:
            A Responder.
        """

    def error_responder(self, exc: Exception) -> 'IWebResponder':
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

    def set_session(self, session: 'IAuthSession'):
        """Attach a session to the requester.

        Args:
            session: A Session object.
        """


class IWebResponder(Protocol):
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


# ----------------------------------------------------------------------------------------------------------------------
# web sites


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


class IWebManager(INode, Protocol):
    """Web manager."""

    sites: list['IWebSite']
    """Configured web sites."""

    def site_from_environ(self, environ: dict) -> 'IWebSite':
        """Returns a site object for the given request environment.

        Args:
            environ: WSGI environment.

        Returns:
            A Site object.
        """


class IWebSite(INode, Protocol):
    """Web site."""

    assetsRoot: Optional[WebDocumentRoot]
    """Root directory for assets."""
    corsOptions: WebCors
    """CORS options."""
    errorPage: Optional['ITemplate']
    """Error page template."""
    host: str
    """Host name for this site."""
    rewriteRules: list[WebRewriteRule]
    """Rewrite rule."""
    staticRoot: WebDocumentRoot
    """Root directory for static files."""

    def url_for(self, req: 'IWebRequester', path: str, **kwargs) -> str:
        """Rewrite a request path to an Url.

        Args:
            req: Web Requester.
            path: Raw request path.
            **kwargs: Extra GET params.

        Returns:
            A rewritten URL.
        """


# ----------------------------------------------------------------------------------------------------------------------
# authorization


class IUser(IObject, Protocol):
    """User object."""

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
    provider: 'IAuthProvider'
    """User authorization provider."""
    roles: set[str]
    """User roles."""
    uid: str
    """Global user uid."""

    def acl_bit(self, access: Access, obj: IObject) -> Optional[int]:
        """Get the ACL bit for a specific object.

        Args:
            access: Access mode.
            obj: Requested object.

        Returns:
            ``1`` or ``0`` if the user's permissions have the bit and ``None`` otherwise.
        """

    def can(self, access: Access, obj: IObject, *context) -> bool:
        """Check if the user can access an object.

        Args:
            access: Access mode.
            obj: Requested object.
            *context: Further objects to check.

        Returns:
            ``True`` is access is granted.
        """

    def can_create(self, obj: IObject, *context) -> bool:
        """Check if the user has "create" permission on an object."""

    def can_delete(self, obj: IObject, *context) -> bool:
        """Check if the user has "delete" permission on an object."""

    def can_read(self, obj: IObject, *context) -> bool:
        """Check if the user has "read" permission on an object."""

    def can_use(self, obj: IObject, *context) -> bool:
        """Check if the user has "read" permission on an object."""

    def can_write(self, obj: IObject, *context) -> bool:
        """Check if the user has "write" permission on an object."""

    def can_edit(self, obj: IObject, *context) -> bool:
        """Check if the user has "edit" permissions on an object."""

    def acquire(self, uid: str, classref: Optional[ClassRef] = None, access: Optional[Access] = None) -> Optional[IObject]:
        """Get a readable object by uid.

        Args:
            uid: Object uid.
            classref: Class reference. If provided, ensures that the object matches the reference.
            access: Access mode, assumed ``Access.read`` if omitted.

        Returns:
            A readable object or ``None`` if the object does not exists or user doesn't have a permission.
        """

    def require(self, uid: str, classref: Optional[ClassRef] = None, access: Optional[Access] = None) -> IObject:
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

    def require_project(self, uid: str) -> 'IProject':
        """Get a readable Project object.

        Args:
            uid: Project uid.

        Returns:
            A Project object.
        """

    def require_layer(self, uid) -> 'ILayer':
        """Get a readable Layer object.

        Args:
            uid: Layer uid.

        Returns:
            A Layer object.
        """


class IAuthManager(INode, Protocol):
    """Authentication manager."""

    guestSession: 'IAuthSession'
    """Preconfigured Guest session."""

    guestUser: 'IUser'
    """Preconfigured Guest user."""
    systemUser: 'IUser'
    """Preconfigured System user."""

    providers: list['IAuthProvider']
    """Authentication providers."""
    methods: list['IAuthMethod']
    """Authentication methods."""
    mfa: list['IAuthMfa']
    """Authentication MFA handlers."""

    sessionMgr: 'IAuthSessionManager'
    """Session manager."""

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']:
        """Authenticate a user.

        Args:
            method: Authentication method.
            credentials: Credentials object.

        Returns:
            An authenticated User or ``None`` if authentication failed.
        """

    def get_user(self, user_uid: str) -> Optional['IUser']:
        """Get a User by its global uid.

        Args:
            user_uid: Global user uid.
        Returns:
            A User or ``None``.
        """

    def get_provider(self, uid: str) -> Optional['IAuthProvider']:
        """Get an authentication Provider by its uid.

        Args:
            uid: Uid.
        Returns:
            A Provider or ``None``.
        """

    def get_method(self, uid: str) -> Optional['IAuthMethod']:
        """Get an authentication Method by its uid.

        Args:
            uid: Uid.
        Returns:
            A Method or ``None``.
        """

    def get_mfa(self, uid: str) -> Optional['IAuthMfa']:
        """Get an authentication Provider by its uid.

        Args:
            uid: Uid.
        Returns:
            A Provider or ``None``.
        """

    def serialize_user(self, user: 'IUser') -> str:
        """Return a string representation of a User.

        Args:
            user: A User object.

        Returns:
            A json string.
        """

    def unserialize_user(self, ser: str) -> Optional['IUser']:
        """Restore a User object from a serialized representation.

        Args:
            ser: A json string.

        Returns:
            A User object.
        """


class IAuthMethod(INode, Protocol):
    """Authentication Method."""

    authMgr: 'IAuthManager'


    secure: bool
    """Method is only allowed in a secure context."""

    def open_session(self, req: IWebRequester) -> Optional['IAuthSession']:
        """Attempt to open a Session for a Requester.

        Args:
            req: Requester object.

        Returns:
            A Session or ``None``.
        """

    def close_session(self, req: IWebRequester, res: IWebResponder) -> bool:
        """Close a previously opened Session.

        Args:
            req: Requester object.
            res: Responder object.

        Returns:
            True if the Session was successfully closed.
        """


class IAuthMfa(INode, Protocol):
    """Authentication MFA handler."""

    autoStart: bool
    lifeTime: int
    maxAttempts: int
    maxRestarts: int

    def start(self, user: 'IUser'): ...

    def is_valid(self, user: 'IUser') -> bool: ...

    def cancel(self, user: 'IUser'): ...

    def verify(self, user: 'IUser', request: Data) -> bool: ...

    def restart(self, user: 'IUser') -> bool: ...


class IAuthProvider(INode, Protocol):
    """Authentication Provider."""

    allowedMethods: list[str]
    """List of Method types allowed to be used with this Provider."""

    def get_user(self, local_uid: str) -> Optional['IUser']:
        """Get a User from its local uid.

        Args:
            local_uid: User local uid.

        Returns:
            A User or ``None``.
        """

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']:
        """Authenticate a user.

        Args:
            method: Authentication method.
            credentials: Credentials object.

        Returns:
            An authenticated User or ``None`` if authentication failed.
        """

    def serialize_user(self, user: 'IUser') -> str:
        """Return a string representation of a User.

        Args:
            user: A User object.

        Returns:
            A json string.
        """

    def unserialize_user(self, ser: str) -> Optional['IUser']:
        """Restore a User object from a serialized representation.

        Args:
            ser: A json string.

        Returns:
            A User object.
        """


class IAuthSession(IObject, Protocol):
    """Authentication session."""

    uid: str
    """Session uid."""
    method: Optional['IAuthMethod']
    """Authentication method that created the session."""
    user: 'IUser'
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


class IAuthSessionManager(INode, Protocol):
    """Authentication session Manager."""

    lifeTime: int
    """Session lifetime in seconds."""

    def create(self, method: 'IAuthMethod', user: 'IUser', data: Optional[dict] = None) -> 'IAuthSession':
        """Create a new Session,

        Args:
            method: Auth Method that creates the Session.
            user: User for which the Session is created.
            data: Session data.

        Returns:
            A new Session.
        """

    def delete(self, sess: 'IAuthSession'):
        """Delete a Session.

        Args:
            sess: Session object.
        """

    def delete_all(self):
        """Delete all Sessions.
        """

    def get(self, uid: str) -> Optional['IAuthSession']:
        """Get Session by its uid.

        Args:
            uid: Session uid.

        Returns:
            A Session or ``None``.
        """

    def get_valid(self, uid: str) -> Optional['IAuthSession']:
        """Get a valid Session by its uid.

        Args:
            uid: Session uid.

        Returns:
            A Session or ``None`` if uid does not exists or the Session is not valid.
        """

    def get_all(self) -> list['IAuthSession']:
        """Get all sessions."""

    def save(self, sess: 'IAuthSession'):
        """Save the Session state into a persistent storage.

        Args:
            sess: Session object.
        """

    def touch(self, sess: 'IAuthSession'):
        """Update the Session last activity timestamp.

        Args:
            sess: Session object.
        """

    def cleanup(self):
        """Remove invalid Sessions from the storage.
        """


# ----------------------------------------------------------------------------------------------------------------------
# attributes

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


# ----------------------------------------------------------------------------------------------------------------------
# CRS

CrsName: TypeAlias = int | str
"""A CRS code like `EPSG:3857` or a srid like `3857`."""


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

    crs: 'ICrs'
    extent: Extent


class ICrs(Protocol):
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

    def transform_extent(self, extent: Extent, crs_to: 'ICrs') -> Extent:
        """Transform an Extent from this CRS to another.

        Args:
            extent: Extent.
            crs_to: Target CRS.

        Returns:
            A transformed Extent.
        """

    def transformer(self, crs_to: 'ICrs') -> Callable:
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


# ----------------------------------------------------------------------------------------------------------------------
# Geodata sources

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
    crs: 'ICrs'
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

    supportedCrs: list['ICrs']
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


# ----------------------------------------------------------------------------------------------------------------------
# XML


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


class IXmlElement(Iterable):
    """XML Element."""

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

    lname: str
    """Element name (tag without a namespace) in lower case."""

    caseInsensitive: bool
    """Element is case-insensitive."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator['IXmlElement']: ...

    def __getitem__(self, item: int) -> 'IXmlElement': ...

    def clear(self): ...

    def get(self, key: str, default=None) -> Any: ...

    def attr(self, key: str, default=None) -> Any: ...

    def items(self) -> Iterable[Any]: ...

    def keys(self) -> Iterable[str]: ...

    def set(self, key: str, value: Any): ...

    def append(self, subelement: 'IXmlElement'): ...

    def extend(self, subelements: Iterable['IXmlElement']): ...

    def insert(self, index: int, subelement: 'IXmlElement'): ...

    def find(self, path: str) -> Optional['IXmlElement']:
        """Finds first matching element by tag name or path."""

    def findall(self, path: str) -> list['IXmlElement']:
        """Finds all matching subelements by name or path."""

    def findtext(self, path: str, default: Optional[str] = None) -> str:
        """Finds text for first matching element by name or path."""

    def iter(self, tag: Optional[str] = None) -> Iterable['IXmlElement']: ...

    def iterfind(self, path: Optional[str] = None) -> Iterable['IXmlElement']:
        """Returns an iterable of all matching subelements by name or path."""

    def itertext(self) -> Iterable[str]: ...

    def remove(self, other: 'IXmlElement'): ...

    # extensions

    def add(self, tag: str, attrib: Optional[dict] = None, **extra) -> 'IXmlElement': ...

    def children(self) -> list['IXmlElement']: ...

    def findfirst(self, *paths) -> Optional['IXmlElement']: ...

    def textof(self, *paths) -> str: ...

    def textlist(self, *paths, deep=False) -> list[str]: ...

    def textdict(self, *paths, deep=False) -> dict[str, str]: ...

    #

    def to_string(
            self,
            compact_whitespace=False,
            remove_namespaces=False,
            with_namespace_declarations=False,
            with_schema_locations=False,
            with_xml_declaration=False,
    ) -> str:
        """Converts the Element object to a string.

        Args:
            compact_whitespace: Remove all whitespace outside of tags and elements.
            remove_namespaces: Remove all namespace references.
            with_namespace_declarations: Include the namespace declarations.
            with_schema_locations: Include schema locations.
            with_xml_declaration: Include the xml declaration.

        Returns:
            An XML string.
        """

    def to_dict(self) -> dict:
        """Creates a dictionary from an XElement object.

        Returns:
            A dict with the keys ``tag``, ``attrib``, ``text``, ``tail``, ``tail``, ``children``.
        """


# ----------------------------------------------------------------------------------------------------------------------
# shapes

class ShapeProps(Props):
    """Shape properties."""

    crs: str
    geometry: dict


class IShape(Protocol):
    """Geo-referenced geometry."""

    type: GeometryType
    """Geometry type."""

    crs: 'ICrs'
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

    def centroid(self) -> 'IShape':
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

    def equals(self, other: 'IShape') -> bool:
        """Returns True if this shape is equal to the other."""

    def contains(self, other: 'IShape') -> bool:
        """Returns True if this shape contains the other."""

    def covers(self, other: 'IShape') -> bool:
        """Returns True if this shape covers the other."""

    def covered_by(self, other: 'IShape') -> bool:
        """Returns True if this shape is covered by the other."""

    def crosses(self, other: 'IShape') -> bool:
        """Returns True if this shape crosses the other."""

    def disjoint(self, other: 'IShape') -> bool:
        """Returns True if this shape does not intersect with the other."""

    def intersects(self, other: 'IShape') -> bool:
        """Returns True if this shape intersects with the other."""

    def overlaps(self, other: 'IShape') -> bool:
        """Returns True if this shape overlaps the other."""

    def touches(self, other: 'IShape') -> bool:
        """Returns True if this shape touches the other."""

    def within(self, other: 'IShape') -> bool:
        """Returns True if this shape is within the other."""

    # set operations

    def union(self, others: list['IShape']) -> 'IShape':
        """Computes a union of this shape and other shapes."""

    def intersection(self, *others: 'IShape') -> 'IShape':
        """Computes an intersection of this shape and other shapes."""

    # convertors

    def to_multi(self) -> 'IShape':
        """Converts a singly-geometry shape to a multi-geometry one."""

    def to_type(self, new_type: 'GeometryType') -> 'IShape':
        """Converts a geometry to another type."""

    # misc

    def tolerance_polygon(self, tolerance, quad_segs=None) -> 'IShape':
        """Builds a buffer polygon around the shape."""

    def transformed_to(self, crs: 'ICrs') -> 'IShape':
        """Returns this shape transformed to another CRS."""


# ----------------------------------------------------------------------------------------------------------------------
# database

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
    """GDAL Dataset description."""

    columns: list[ColumnDescription]
    columnMap: dict[str, ColumnDescription]
    fullName: str
    geometryName: str
    geometrySrid: int
    geometryType: GeometryType
    name: str
    schema: str


class IDatabaseManager(INode, Protocol):
    """Database manager."""

    def create_provider(self, cfg: Config, **kwargs) -> 'IDatabaseProvider': ...

    def providers(self) -> list['IDatabaseProvider']: ...

    def provider(self, uid: str) -> Optional['IDatabaseProvider']: ...

    def first_provider(self, ext_type: str) -> Optional['IDatabaseProvider']: ...


class IDatabaseProvider(IProvider, Protocol):
    """Database Provider."""

    mgr: 'IDatabaseManager'
    url: str
    models: list['IDatabaseModel']

    def connection(self) -> 'sqlalchemy.Connection': ...

    def engine(self, **kwargs) -> 'sqlalchemy.Engine': ...

    def split_table_name(self, table_name: str) -> tuple[str, str]: ...

    def join_table_name(self, schema: str, name: str) -> str: ...

    def table(self, table_name: str, **kwargs) -> 'sqlalchemy.Table': ...

    def has_table(self, table_name: str) -> bool: ...

    def column(self, table: 'sqlalchemy.Table', column_name: str) -> 'sqlalchemy.Column': ...

    def has_column(self, table: 'sqlalchemy.Table', column_name: str) -> bool: ...

    def describe(self, table_name: str) -> DataSetDescription: ...

    def table_bounds(self, table_name) -> Optional[Bounds]: ...


# ----------------------------------------------------------------------------------------------------------------------
# storage

class IStorageManager(INode, Protocol):
    """Storage manager."""

    def provider(self, uid: str) -> Optional['IStorageProvider']: ...

    def first_provider(self) -> Optional['IStorageProvider']: ...


class StorageRecord(Data):
    """Storage record."""

    name: str
    userUid: str
    data: str
    created: int
    updated: int


class IStorageProvider(INode, Protocol):
    """Storage provider."""

    def list_names(self, category: str) -> list[str]: ...

    def read(self, category: str, name: str) -> Optional['StorageRecord']: ...

    def write(self, category: str, name: str, data: str, user_uid: str): ...

    def delete(self, category: str, name: str): ...


# ----------------------------------------------------------------------------------------------------------------------
# features


FeatureUid: TypeAlias = str
"""Unique Feature id."""

class FeatureRecord(Data):
    """Raw data from a feature source."""

    attributes: dict
    meta: dict
    uid: Optional[str]
    shape: Optional['IShape']


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


class IFeature(Protocol):
    """Feature object."""

    attributes: dict
    category: str
    cssSelector: str
    errors: list['ModelValidationError']
    isNew: bool
    model: 'IModel'
    props: 'FeatureProps'
    record: 'FeatureRecord'
    views: dict
    createWithFeatures: list['IFeature']
    insertedPrimaryKey: str

    def get(self, name: str, default=None) -> Any: ...

    def has(self, name: str) -> bool: ...

    def set(self, name: str, value: Any) -> 'IFeature': ...

    def raw(self, name: str) -> Any: ...

    def render_views(self, templates: list['ITemplate'], **kwargs) -> 'IFeature': ...

    def shape(self) -> Optional['IShape']: ...

    def to_geojson(self, user: 'IUser') -> dict: ...

    def to_svg(self, view: 'MapView', label: Optional[str] = None, style: Optional['IStyle'] = None) -> list[IXmlElement]: ...

    def transform_to(self, crs: 'ICrs') -> 'IFeature': ...

    def uid(self) -> FeatureUid: ...


# ----------------------------------------------------------------------------------------------------------------------
# models

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
    user: 'IUser'
    project: 'IProject'
    relDepth: int = 0
    maxDepth: int = 0
    search: 'SearchQuery'
    dbSelect: ModelDbSelect
    dbConnection: 'sqlalchemy.Connection'


EmptyValue = object()
"""Special value for empty fields."""

ErrorValue = object()
"""Special value for invalid fields."""


class IModelWidget(INode, Protocol):
    """Model widget."""

    supportsTableView: bool = True


class IModelValidator(INode, Protocol):
    """Model Validator."""

    message: str
    ops: set[ModelOperation]

    def validate(self, field: 'IModelField', feature: 'IFeature', mc: ModelContext) -> bool: ...


class IModelValue(INode, Protocol):
    """Model value."""

    isDefault: bool
    ops: set[ModelOperation]

    def compute(self, field: 'IModelField', feature: 'IFeature', mc: 'ModelContext'): ...


class IModelField(INode, Protocol):
    """Model field."""

    name: str
    title: str

    attributeType: AttributeType

    widget: Optional['IModelWidget'] = None

    values: list['IModelValue']
    validators: list['IModelValidator']

    isPrimaryKey: bool
    isRequired: bool
    isUnique: bool
    isAuto: bool

    supportsFilterSearch: bool = False
    supportsGeometrySearch: bool = False
    supportsKeywordSearch: bool = False

    model: 'IModel'

    def before_select(self, mc: ModelContext): ...

    def after_select(self, features: list['IFeature'], mc: ModelContext): ...

    def before_create(self, feature: 'IFeature', mc: ModelContext): ...

    def after_create(self, feature: 'IFeature', mc: ModelContext): ...

    def before_create_related(self, to_feature: 'IFeature', mc: ModelContext): ...

    def after_create_related(self, to_feature: 'IFeature', mc: ModelContext): ...

    def before_update(self, feature: 'IFeature', mc: ModelContext): ...

    def after_update(self, feature: 'IFeature', mc: ModelContext): ...

    def before_delete(self, feature: 'IFeature', mc: ModelContext): ...

    def after_delete(self, feature: 'IFeature', mc: ModelContext): ...

    def do_init(self, feature: 'IFeature', mc: ModelContext): ...

    def do_init_related(self, to_feature: 'IFeature', mc: ModelContext): ...

    def do_validate(self, feature: 'IFeature', mc: ModelContext): ...

    def from_props(self, feature: 'IFeature', mc: ModelContext): ...

    def to_props(self, feature: 'IFeature', mc: ModelContext): ...

    def from_record(self, feature: 'IFeature', mc: ModelContext): ...

    def to_record(self, feature: 'IFeature', mc: ModelContext): ...

    def related_models(self) -> list['IModel']: ...

    def find_relatable_features(self, search: 'SearchQuery', mc: ModelContext) -> list['IFeature']: ...

    def raw_to_python(self, feature: 'IFeature', value, mc: ModelContext): ...

    def prop_to_python(self, feature: 'IFeature', value, mc: ModelContext): ...

    def python_to_raw(self, feature: 'IFeature', value, mc: ModelContext): ...

    def python_to_prop(self, feature: 'IFeature', value, mc: ModelContext): ...

    def describe(self) -> Optional[ColumnDescription]: ...


class IModel(INode, Protocol):
    """Data Model."""

    defaultSort: list['SearchSort']
    fields: list['IModelField']
    geometryCrs: Optional['ICrs']
    geometryName: str
    geometryType: Optional[GeometryType]
    isEditable: bool
    loadingStrategy: 'FeatureLoadingStrategy'
    title: str
    uidName: str
    withTableView: bool

    def find_features(self, search: 'SearchQuery', mc: ModelContext) -> list['IFeature']: ...

    def get_features(self, uids: Iterable[str | int], mc: ModelContext) -> list['IFeature']: ...

    def init_feature(self, feature: 'IFeature', mc: ModelContext): ...

    def create_feature(self, feature: 'IFeature', mc: ModelContext) -> FeatureUid: ...

    def update_feature(self, feature: 'IFeature', mc: ModelContext) -> FeatureUid: ...

    def delete_feature(self, feature: 'IFeature', mc: ModelContext) -> FeatureUid: ...

    def validate_feature(self, feature: 'IFeature', mc: ModelContext) -> bool: ...

    def feature_from_props(self, props: 'FeatureProps', mc: ModelContext) -> 'IFeature': ...

    def feature_to_props(self, feature: 'IFeature', mc: ModelContext) -> 'FeatureProps': ...

    def feature_to_view_props(self, feature: 'IFeature', mc: ModelContext) -> 'FeatureProps': ...

    def describe(self) -> Optional[DataSetDescription]: ...

    def field(self, name: str) -> Optional['IModelField']: ...

    def related_models(self) -> list['IModel']: ...


class IDatabaseModel(IModel, Protocol):
    """Database-based data model."""

    provider: 'IDatabaseProvider'
    sqlFilter: str
    tableName: str

    def table(self) -> 'sqlalchemy.Table': ...

    def column(self, column_name: str) -> 'sqlalchemy.Column': ...

    def uid_column(self) -> 'sqlalchemy.Column': ...

    def connection(self) -> 'sqlalchemy.Connection': ...

    def execute(self, sql: 'sqlalchemy.Executable', mc: ModelContext, parameters=None) -> 'sqlalchemy.CursorResult': ...


class IModelManager(INode, Protocol):
    """Model manager."""

    def get_model(self, uid: str, user: IUser = None, access: Access = None) -> Optional['IModel']: ...

    def locate_model(self, *objects, user: IUser = None, access: Access = None) -> Optional['IModel']: ...

    def editable_models(self, project: 'IProject', user: 'IUser') -> list['IModel']: ...

    def default_model(self) -> 'IModel': ...


# ----------------------------------------------------------------------------------------------------------------------
# templates and render

class ImageFormat(Enum):
    """Image format"""

    png8 = 'png8'
    """png 8-bit"""
    png24 = 'png24'
    """png 24-bit"""


class IImage(IObject, Protocol):
    """Image object."""

    def size(self) -> Size: ...

    def add_box(self, color=None) -> 'IImage': ...

    def add_text(self, text: str, x=0, y=0, color=None) -> 'IImage': ...

    def compose(self, other: 'IImage', opacity=1) -> 'IImage': ...

    def crop(self, box) -> 'IImage': ...

    def paste(self, other: 'IImage', where=None) -> 'IImage': ...

    def resize(self, size: Size, **kwargs) -> 'IImage': ...

    def rotate(self, angle: int, **kwargs) -> 'IImage': ...

    def to_bytes(self, mime: Optional[str] = None) -> bytes: ...

    def to_path(self, path: str, mime: Optional[str] = None) -> str: ...

    def to_array(self) -> 'numpy.typing.NDArray': ...


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
    features: list['IFeature']
    image: 'IImage'
    layer: 'ILayer'
    opacity: float
    soupPoints: list[Point]
    soupTags: list[Any]
    styles: list['IStyle']
    subLayers: list[str]


class MapRenderInput(Data):
    """Map render input."""

    backgroundColor: int
    bbox: Extent
    center: Point
    crs: 'ICrs'
    dpi: int
    mapSize: MSize
    notify: Callable
    planes: list['MapRenderInputPlane']
    project: 'IProject'
    rotation: int
    scale: int
    user: 'IUser'
    visibleLayers: Optional[list['ILayer']]


class MapRenderOutputPlaneType(Enum):
    """Map render output plane type."""

    image = 'image'
    path = 'path'
    svg = 'svg'


class MapRenderOutputPlane(Data):
    """Map render output plane."""

    type: MapRenderOutputPlaneType
    path: str
    elements: list[IXmlElement]
    image: 'IImage'


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
    project: 'IProject'
    style: 'IStyle'
    type: LayerRenderInputType
    user: 'IUser'
    view: MapView
    x: int
    y: int
    z: int


class LayerRenderOutput(Data):
    """Layer render output."""

    content: bytes
    tags: list[IXmlElement]


class TemplateRenderInput(Data):
    """Template render input."""

    args: dict
    crs: ICrs
    dpi: int
    localeUid: str
    maps: list[MapRenderInput]
    mimeOut: str
    notify: Callable
    project: 'IProject'
    user: 'IUser'


class TemplateQualityLevel(Data):
    """Template quality level."""

    name: str
    dpi: int


class ITemplate(INode, Protocol):
    """Template object."""

    mapSize: MSize
    mimeTypes: list[str]
    pageMargin: MExtent
    pageSize: MSize
    subject: str
    title: str

    def render(self, tri: TemplateRenderInput) -> ContentResponse: ...


class ITemplateManager(INode, Protocol):
    """Template manager."""

    def find_template(self, *objects, user: IUser = None, subject: str = None, mime: str = None) -> Optional['ITemplate']: ...

    def template_from_path(self, path: str) -> Optional['ITemplate']: ...


# ----------------------------------------------------------------------------------------------------------------------
# jobs


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


class IJob(Protocol):
    """Background Job object."""

    error: str
    payload: dict
    state: JobState
    uid: str
    user: IUser

    def run(self): ...

    def update(self, payload: Optional[dict] = None, state: Optional[JobState] = None, error: Optional[str] = None): ...

    def cancel(self): ...

    def remove(self): ...


# ----------------------------------------------------------------------------------------------------------------------
# printing


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
    state: JobState
    stepType: str
    stepName: str
    url: str


class IPrinter(INode, Protocol):
    """Printer object."""

    title: str
    template: 'ITemplate'
    models: list['IModel']
    qualityLevels: list[TemplateQualityLevel]


class IPrinterManager(INode, Protocol):
    """Print Manager."""

    def printers_for_project(self, project: 'IProject', user: 'IUser') -> list['IPrinter']: ...

    def start_job(self, request: PrintRequest, user: IUser) -> IJob: ...

    def get_job(self, uid: str, user: IUser) -> Optional[IJob]: ...

    def run_job(self, request: PrintRequest, user: IUser): ...

    def cancel_job(self, job: IJob): ...

    def result_path(self, job: IJob) -> str: ...

    def status(self, job: IJob) -> PrintJobResponse: ...


# ----------------------------------------------------------------------------------------------------------------------
# styles

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


class IStyle(IObject, Protocol):
    """CSS Style object."""

    cssSelector: str
    text: str
    values: StyleValues


# ----------------------------------------------------------------------------------------------------------------------
# locale

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


# ----------------------------------------------------------------------------------------------------------------------
# metadata


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


# ----------------------------------------------------------------------------------------------------------------------
# search


class SearchSort(Data):
    """Search sort specification."""

    fieldName: str
    reverse: bool


class SearchOgcFilter(Data):
    """Search filter."""

    name: str
    operator: str
    shape: 'IShape'
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
    layers: list['ILayer']
    limit: int
    ogcFilter: SearchOgcFilter
    project: 'IProject'
    relDepth: int
    resolution: float
    shape: 'IShape'
    sort: list[SearchSort]
    tolerance: 'Measurement'
    uids: list[str]


class SearchResult(Data):
    """Search result."""

    feature: 'IFeature'
    layer: 'ILayer'
    finder: 'IFinder'


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


class ISearchManager(INode, Protocol):
    """Search Manager."""

    def run_search(self, search: 'SearchQuery', user: IUser) -> list['SearchResult']: ...


class IFinder(INode, Protocol):
    """Finder object."""

    title: str

    supportsFilterSearch: bool = False
    supportsGeometrySearch: bool = False
    supportsKeywordSearch: bool = False

    withFilter: bool
    withGeometry: bool
    withKeyword: bool

    templates: list['ITemplate']
    models: list['IModel']

    provider: 'IProvider'
    sourceLayers: list['SourceLayer']

    tolerance: 'Measurement'

    def run(self, search: SearchQuery, user: IUser, layer: Optional['ILayer'] = None) -> list['IFeature']: ...

    def can_run(self, search: SearchQuery, user: IUser) -> bool: ...


# ----------------------------------------------------------------------------------------------------------------------
# maps and layers


class IMap(INode, Protocol):
    """Map object."""

    rootLayer: 'ILayer'

    bounds: Bounds
    center: Point
    coordinatePrecision: int
    initResolution: float
    resolutions: list[float]
    title: str
    wgsExtent: Extent


class LegendRenderOutput(Data):
    """Legend render output."""

    html: str
    image: 'IImage'
    image_path: str
    size: Size
    mime: str


class ILegend(INode, Protocol):
    """Legend object."""

    def render(self, args: Optional[dict] = None) -> Optional[LegendRenderOutput]: ...


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

    enabled: bool
    layerName: str
    featureName: str
    xmlNamespace: 'XmlNamespace'
    geometryName: str


class ILayer(INode, Protocol):
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
    mapCrs: 'ICrs'
    clientOptions: LayerClientOptions
    displayMode: LayerDisplayMode
    loadingStrategy: FeatureLoadingStrategy
    imageFormat: str
    opacity: float
    resolutions: list[float]
    title: str

    owsOptions: Optional['LayerOwsOptions']

    grid: Optional[TileGrid]
    cache: Optional[LayerCache]

    metadata: 'Metadata'
    legend: Optional['ILegend']
    legendUrl: str

    finders: list['IFinder']
    templates: list['ITemplate']
    models: list['IModel']

    layers: list['ILayer']

    provider: 'IProvider'
    sourceLayers: list['SourceLayer']

    def ancestors(self) -> list['ILayer']: ...

    def descendants(self) -> list['ILayer']: ...

    def render(self, lri: LayerRenderInput) -> Optional['LayerRenderOutput']: ...

    def get_features_for_view(self, search: SearchQuery, user: 'IUser', view_names: Optional[list[str]] = None) -> list['IFeature']: ...

    def render_legend(self, args: Optional[dict] = None) -> Optional['LegendRenderOutput']: ...

    def url_path(self, kind: str) -> str: ...


# ----------------------------------------------------------------------------------------------------------------------
# OWS

class OwsProtocol(Enum):
    """Supported OWS protocol."""

    WMS = 'WMS'
    WMTS = 'WMTS'
    WCS = 'WCS'
    WFS = 'WFS'
    CSW = 'CSW'


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


class IOwsService(INode, Protocol):
    """OWS Service."""

    isRasterService: bool
    isVectorService: bool

    metadata: 'Metadata'
    name: str
    protocol: OwsProtocol
    supportedBounds: list[Bounds]
    supportedVersions: list[str]
    supportedOperations: list['OwsOperation']
    templates: list['ITemplate']
    updateSequence: str
    version: str
    withInspireMeta: bool
    withStrictParams: bool

    def handle_request(self, req: 'IWebRequester') -> ContentResponse: ...


class IOwsProvider(INode, Protocol):
    """OWS services Provider."""

    alwaysXY: bool
    forceCrs: 'ICrs'
    maxRequests: int
    metadata: 'Metadata'
    operations: list[OwsOperation]
    protocol: OwsProtocol
    sourceLayers: list['SourceLayer']
    url: Url
    version: str

    def get_operation(self, verb: OwsVerb, method: Optional[RequestMethod] = None) -> Optional[OwsOperation]: ...

    def get_features(self, args: SearchQuery, source_layers: list[SourceLayer]) -> list[FeatureRecord]: ...


# ----------------------------------------------------------------------------------------------------------------------


# CLI

class CliParams(Data):
    """CLI params"""
    pass


# ----------------------------------------------------------------------------------------------------------------------
# actions and apis


class IActionManager(INode, Protocol):
    """Action manager."""

    def actions_for_project(self, project: 'IProject', user: IUser) -> list['IAction']:
        """Get a list of actions for a Project, to which a User has access to."""

    def find_action(self, project: Optional['IProject'], ext_type: str, user: IUser) -> Optional['IAction']:
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
            user: 'IUser',
            read_options=None,
    ) -> tuple[Callable, Request]: ...


class IAction(INode, Protocol):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# projects


class IClient(INode, Protocol):
    """GWS Client control object."""

    options: dict
    elements: list


class IProject(INode, Protocol):
    """Project object."""

    assetsRoot: Optional['WebDocumentRoot']
    client: 'IClient'

    localeUids: list[str]
    map: 'IMap'
    metadata: 'Metadata'

    actions: list['IAction']
    finders: list['IFinder']
    models: list['IModel']
    printers: list['IPrinter']
    templates: list['ITemplate']
    owsServices: list['IOwsService']


# ----------------------------------------------------------------------------------------------------------------------
# application

class IMonitor(INode, Protocol):
    """File Monitor facility."""

    def add_directory(self, path: str, pattern: Regex):
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


class IMiddlewareManager(INode, Protocol):
    def register(self, obj: INode, name: str, depends_on: Optional[list[str]] = None):
        """Register an object as a middleware."""

    def objects(self) -> list[INode]:
        """Return a list of registered middleware objects."""


class IApplication(INode, Protocol):
    """The main Application object."""

    client: 'IClient'
    localeUids: list[str]
    metadata: 'Metadata'
    monitor: 'IMonitor'
    version: str
    versionString: str
    defaultPrinter: 'IPrinter'

    actionMgr: 'IActionManager'
    authMgr: 'IAuthManager'
    databaseMgr: 'IDatabaseManager'
    modelMgr: 'IModelManager'
    printerMgr: 'IPrinterManager'
    searchMgr: 'ISearchManager'
    storageMgr: 'IStorageManager'
    templateMgr: 'ITemplateManager'
    webMgr: 'IWebManager'
    middlewareMgr: 'IMiddlewareManager'

    actions: list['IAction']
    projects: list['IProject']
    finders: list['IFinder']
    templates: list['ITemplate']
    printers: list['IPrinter']
    models: list['IModel']
    owsServices: list['IOwsService']

    def project(self, uid: str) -> Optional['IProject']:
        """Get a Project object by its uid."""

    def helper(self, ext_type: str) -> Optional['INode']:
        """Get a Helper object by its extension type."""

    def developer_option(self, key: str):
        """Get a value of a developer option."""
