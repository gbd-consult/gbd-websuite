"""Data classes for the QFieldCloud API, based on its ``swagger.yaml``."""

from datetime import datetime
from typing import Optional

import gws

class JobStatusEnum(gws.Enum):
    """Status of a job."""

    pending = 'pending'
    queued = 'queued'
    started = 'started'
    finished = 'finished'
    stopped = 'stopped'
    failed = 'failed'


class LastStatusEnum(gws.Enum):
    """Last status of a delta."""

    pending = 'pending'
    started = 'started'
    applied = 'applied'
    conflict = 'conflict'
    not_applied = 'not_applied'
    error = 'error'
    ignored = 'ignored'
    unpermitted = 'unpermitted'


class OrganizationMemberRoleEnum(gws.Enum):
    """Role of an organization member."""

    admin = 'admin'
    member = 'member'


class ProjectCollaboratorRoleEnum(gws.Enum):
    """Role of a project collaborator."""

    admin = 'admin'
    manager = 'manager'
    editor = 'editor'
    reporter = 'reporter'
    reader = 'reader'


class ProjectStatusEnum(gws.Enum):
    """Status of a project."""

    ok = 'ok'
    busy = 'busy'
    failed = 'failed'


class TypeEnum(gws.Enum):
    """Type of a job."""

    package = 'package'
    delta_apply = 'delta_apply'
    process_projectfile = 'process_projectfile'


# Data Classes


class Login(gws.Data):
    """Login credentials."""

    password: str
    """Password."""
    username: Optional[str]
    """User name."""
    email: Optional[str]
    """Email address."""


class CompleteUser(gws.Data):
    """Full information about the authenticated user."""

    username: str
    """User name."""
    type: int
    """User type, see ``UserType``."""
    full_name: str
    """Full name."""
    avatar_url: str
    """URL of the avatar image."""
    email: Optional[str]
    """Email address."""
    first_name: Optional[str]
    """First name."""
    last_name: Optional[str]
    """Last name."""


class PublicInfoUser(gws.Data):
    """Public information about a user."""

    username: str
    """User name."""
    type: int
    """User type, see ``UserType``."""
    full_name: str
    """Full name."""
    avatar_url: str
    """URL of the avatar image."""
    username_display: str
    """User name for display."""


class DeltaFeature(gws.Data):
    """Feature state in a delta, before or after the change."""

    attributes: dict
    """Feature attributes."""
    geometry: Optional[str]
    """Geometry as WKT, only present if the geometry has changed."""
    files_sha256: Optional[dict]
    """Checksums of attached files by file name."""


class Delta(gws.Data):
    """A single change, as sent by QField in a delta payload."""

    uuid: str
    """Delta id."""
    clientId: str
    """Id of the local export on the device."""
    exportId: str
    """Id of the package export."""
    localLayerId: str
    """Id of the layer in the packaged QGIS project."""
    localLayerName: str
    """Name of the layer in the packaged QGIS project."""
    localLayerCrs: str
    """CRS of the layer in the packaged QGIS project."""
    localPk: str
    """Primary key of the feature in the packaged layer."""
    sourceLayerId: str
    """Id of the layer in the source QGIS project."""
    sourcePk: str
    """Primary key of the feature in the source layer."""
    method: str
    """Change type, ``create``, ``patch`` or ``delete``."""
    new: Optional[DeltaFeature]
    """Feature after the change, ``None`` for deletions."""
    old: Optional[DeltaFeature]
    """Feature before the change, ``None`` for creations."""


class DeltaStatusType(gws.Enum):
    """Status of a delta."""

    pending = 'STATUS_PENDING'
    busy = 'STATUS_BUSY'
    applied = 'STATUS_APPLIED'
    conflict = 'STATUS_CONFLICT'
    not_applied = 'STATUS_NOT_APPLIED'
    error = 'STATUS_ERROR'
    ignored = 'STATUS_IGNORED'
    unpermitted = 'STATUS_UNPERMITTED'


class Job(gws.Data):
    """A server job, e.g. packaging."""

    type: TypeEnum
    """Job type."""
    id: str
    """Job id."""
    created_at: datetime
    """Creation time."""
    created_by: int
    """Id of the user who created the job."""
    project_id: str
    """Project id."""
    status: JobStatusEnum
    """Job status."""
    updated_at: datetime
    """Last update time."""
    started_at: Optional[datetime]
    """Start time."""
    finished_at: Optional[datetime]
    """Finish time."""


class Organization(gws.Data):
    """An organization."""

    username: str
    """Organization name."""
    type: int
    """User type, see ``UserType``."""
    avatar_url: str
    """URL of the avatar image."""
    members: str
    """Organization members."""
    organization_owner: str
    """Owner of the organization."""
    membership_role: str
    """Role of the current user in the organization."""
    membership_role_origin: str
    """Origin of the membership role."""
    membership_is_public: bool
    """Whether the membership is public."""
    teams: list[str]
    """Organization teams."""
    email: Optional[str]
    """Email address."""


class OrganizationMember(gws.Data):
    """A member of an organization."""

    organization: str
    """Organization name."""
    member: str
    """Member user name."""
    role: OrganizationMemberRoleEnum
    """Member role."""
    is_public: Optional[bool]
    """Whether the membership is public."""


class Project(gws.Data):
    """A QField Cloud project."""

    id: str
    """Project id."""
    name: str
    """Project name."""
    owner: str
    """Project owner."""
    created_at: datetime
    """Creation time."""
    updated_at: datetime
    """Last update time."""
    can_repackage: bool
    """Whether the project can be packaged again."""
    needs_repackaging: bool
    """Whether the project must be packaged again."""
    status: ProjectStatusEnum
    """Project status."""
    user_role: str
    """Role of the current user in the project."""
    user_role_origin: str
    """Origin of the user role."""
    is_shared_datasets_project: bool
    """Whether this is a shared datasets project."""
    is_attachment_download_on_demand: bool
    """Whether attachments are downloaded on demand."""
    description: Optional[str]
    """Project description."""
    private: Optional[bool]
    """Whether the project is private."""
    is_public: Optional[bool]
    """Whether the project is public."""
    data_last_packaged_at: Optional[datetime]
    """Time of the last packaging."""
    data_last_updated_at: Optional[datetime]
    """Time of the last data update."""
    shared_datasets_project_id: Optional[str]
    """Id of the shared datasets project."""
    is_featured: Optional[bool]
    """Whether the project is featured."""


class ProjectCollaborator(gws.Data):
    """A collaborator of a project."""

    collaborator: str
    """Collaborator user name."""
    project_id: str
    """Project id."""
    created_by: str
    """User who added the collaborator."""
    updated_by: str
    """User who last updated the collaborator."""
    created_at: datetime
    """Creation time."""
    updated_at: datetime
    """Last update time."""
    role: Optional[ProjectCollaboratorRoleEnum]
    """Collaborator role."""


class Team(gws.Data):
    """A team in an organization."""

    team: str
    """Team name."""
    organization: str
    """Organization name."""
    members: list[str]
    """Team members."""


class TeamMember(gws.Data):
    """A member of a team."""

    member: str
    """Member user name."""


## not in swagger.yaml but used in code


class Package(gws.Data):
    """A project package, as listed for download."""

    files: list[dict]
    """Package files."""
    layers: list[dict]
    """Package layers."""
    status: JobStatusEnum
    """Packaging status."""
    package_id: str
    """Package id."""
    packaged_at: datetime
    """Packaging time."""
    data_last_updated_at: datetime
    """Time of the last data update."""


class AuthProvider(gws.Data):
    """An authentication provider offered to the client."""

    type: str
    """Provider type."""
    id: str
    """Provider id."""
    name: str
    """Provider name for display."""


class UserType(gws.Enum):
    """Type of a user account."""

    person = 1
    organization = 2
    team = 3


class AuthToken(gws.Data):
    """Authentication token and user information returned on login."""

    token: str
    """Session token."""
    expires_at: datetime
    """Expiration time."""
    username: str
    """User name."""
    type: UserType
    """User type."""
    full_name: str
    """Full name."""
    avatar_url: str
    """URL of the avatar image."""
    email: Optional[str]
    """Email address."""
    first_name: Optional[str]
    """First name."""
    last_name: Optional[str]
    """Last name."""


class ServerInfo(gws.Data):
    """Server information."""

    version: str
    """Server version."""
    auth_providers: list[AuthProvider]
    """Available authentication providers."""
    signup_url: str
    """URL of the signup page."""
    whitelabel: dict
    """Branding options."""


class Status(gws.Data):
    """Server status."""

    version: str
    """Server version."""
    database: str
    """Database status."""
    storage: str
    """Storage status."""
    status_page_url: Optional[str]
    """URL of the status page."""
    incident_message: Optional[str]
    """Incident message."""
    incident_timestamp_utc: Optional[str]
    """Incident time."""
    maintenance_message: Optional[str]
    """Maintenance message."""
    maintenance_start_timestamp_utc: Optional[str]
    """Maintenance start time."""
    maintenance_end_timestamp_utc: Optional[str]
    """Maintenance end time."""


class Subscription(gws.Data):
    """Subscription of a user."""

    plan_display_name: str
    """Plan name for display."""
    active_storage_total_bytes: int
    """Total storage available, in bytes."""
    storage_used_bytes: int
    """Storage used, in bytes."""
    plan_storage_threshold_warning_bytes: int
    """Storage warning threshold, in bytes."""
    plan_storage_threshold_critical_bytes: int
    """Storage critical threshold, in bytes."""
    status: str
    """Subscription status."""


class PostJobPayload(gws.Data):
    """Payload of a job creation request."""

    type: TypeEnum
    """Job type."""
    project_id: str
    """Project id."""


class DeltasPayload(gws.Data):
    """A delta payload (delta file) sent by QField."""

    deltas: list[Delta]
    """List of deltas."""
    files: list[dict]
    """List of files."""
    id: str
    """Payload id."""
    project: str
    """Project id."""
    version: str
    """Delta file format version."""


class PackageFile(gws.Data):
    """A file in a package."""

    name: str
    """File name, relative to the package."""
    size: int
    """File size in bytes."""
    uploaded_at: str
    """Upload time."""
    is_attachment: bool
    """Whether the file is an attachment."""
    md5sum: str
    """MD5 checksum, in the format QField expects."""
    last_modified: str
    """Modification time."""
    sha256: str
    """SHA-256 checksum."""
