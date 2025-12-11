"""API definition from swagger.yaml"""

import inspect
from datetime import datetime
from typing import Any, List, Optional, TypeVar, get_args, get_origin

import gws

class JobStatusEnum(gws.Enum):
    pending = 'pending'
    queued = 'queued'
    started = 'started'
    finished = 'finished'
    stopped = 'stopped'
    failed = 'failed'


class LastStatusEnum(gws.Enum):
    pending = 'pending'
    started = 'started'
    applied = 'applied'
    conflict = 'conflict'
    not_applied = 'not_applied'
    error = 'error'
    ignored = 'ignored'
    unpermitted = 'unpermitted'


class OrganizationMemberRoleEnum(gws.Enum):
    admin = 'admin'
    member = 'member'


class ProjectCollaboratorRoleEnum(gws.Enum):
    admin = 'admin'
    manager = 'manager'
    editor = 'editor'
    reporter = 'reporter'
    reader = 'reader'


class ProjectStatusEnum(gws.Enum):
    ok = 'ok'
    busy = 'busy'
    failed = 'failed'


class TypeEnum(gws.Enum):
    package = 'package'
    delta_apply = 'delta_apply'
    process_projectfile = 'process_projectfile'


# Data Classes


class Login(gws.Data):
    password: str
    username: Optional[str]
    email: Optional[str]


class CompleteUser(gws.Data):
    username: str
    type: int
    full_name: str
    avatar_url: str
    email: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]


class PublicInfoUser(gws.Data):
    username: str
    type: int
    full_name: str
    avatar_url: str
    username_display: str


class Delta(gws.Data):
    deltafile_id: str
    content: Any
    id: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    status: str
    output: str
    last_status: Optional[LastStatusEnum]
    last_feedback: Optional[Any]


class Job(gws.Data):
    type: TypeEnum
    id: str
    created_at: datetime
    created_by: int
    project_id: str
    status: JobStatusEnum
    updated_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class Organization(gws.Data):
    username: str
    type: int
    avatar_url: str
    members: str
    organization_owner: str
    membership_role: str
    membership_role_origin: str
    membership_is_public: bool
    teams: List[str]
    email: Optional[str]


class OrganizationMember(gws.Data):
    organization: str
    member: str
    role: OrganizationMemberRoleEnum
    is_public: Optional[bool]


class Project(gws.Data):
    id: str
    name: str
    owner: str
    created_at: datetime
    updated_at: datetime
    can_repackage: bool
    needs_repackaging: bool
    status: ProjectStatusEnum
    user_role: str
    user_role_origin: str
    is_shared_datasets_project: bool
    is_attachment_download_on_demand: bool
    description: Optional[str]
    private: Optional[bool]
    is_public: Optional[bool]
    data_last_packaged_at: Optional[datetime]
    data_last_updated_at: Optional[datetime]
    shared_datasets_project_id: Optional[str]
    is_featured: Optional[bool]


class ProjectCollaborator(gws.Data):
    collaborator: str
    project_id: str
    created_by: str
    updated_by: str
    created_at: datetime
    updated_at: datetime
    role: Optional[ProjectCollaboratorRoleEnum]


class Team(gws.Data):
    team: str
    organization: str
    members: List[str]


class TeamMember(gws.Data):
    member: str


## not in swagger.yaml but used in code


class Package(gws.Data):
    files: List[dict]
    layers: List[dict]
    status: JobStatusEnum
    package_id: str
    packaged_at: datetime
    data_last_updated_at: datetime


class AuthProvider(gws.Data):
    type: str
    id: str
    name: str


class UserType(gws.Enum):
    person = 1
    organization = 2
    team = 3


class AuthToken(gws.Data):
    token: str
    expires_at: datetime
    username: str
    type: UserType
    full_name: str
    avatar_url: str
    email: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]


class PostJobPayload(gws.Data):
    type: TypeEnum
    project_id: str


class DeltasPayload(gws.Data):
    deltas: list[dict]
    files: list[dict]
    id: str
    project: str
    version: str


class StoredDelta(gws.Data):
    id: str
    deltafile_id: str
    created_by: str
    created_at: str
    updated_at: str
    status: str
    client_id: str
    output: Optional[dict]
    last_status: str
    last_feedback: Optional[dict]
    content: dict


class PackageFile(gws.Data):
    name: str
    size: int
    uploaded_at: str
    is_attachment: bool
    md5sum: str
    last_modified: str
    sha256: str
