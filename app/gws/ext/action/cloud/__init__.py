import os

import gws
import gws.config
import gws.server
import gws.web.error
import gws.tools.mime
import gws.tools.job

import gws.types as t
import gws.common.template


class Config(t.WithTypeAndAccess):
    """Cloud admin action"""
    pass


class DataSet(t.Data):
    """Data set for a cloud project."""
    uid: str  #: data set uid as used in the map
    records: t.List[dict]  #: list of data records


class AssetType(t.Enum):
    svg = "svg"


class Source(t.Data):
    text: t.Optional[str]  #: text content
    content: t.Optional[bytes]  #: binary content


class Asset(t.Data):
    """Media asset for the map"""

    type: AssetType  #: type, e.g. "svg"
    name: str  #: file name, as used in the map source
    source: Source #: file source


class MapType(t.Enum):
    qgis = "qgis"


class Map(t.Data):
    """Cloud project map"""

    type: MapType  #: map type
    source: Source  #: map source code (e.g. a QGIS project)
    content: t.Optional[bytes]  #: binary file content
    layerUids: t.Optional[t.List[str]]  #: layer ids to use in the cloud
    data: t.Optional[t.List[DataSet]]  #: data for the map
    assets: t.Optional[t.List[Asset]]  #: map assets


class CreateProjectParams(t.Params):
    """Parameters for the CreateProject command"""

    userUid: str  #: API user ID
    userKey: str  #: API user key
    projectName: str  #: project name
    map: Map  #: project map


class CreateProjectResponse(t.Response):
    pass


class Object(gws.ActionObject):

    def api_create_project(self, req, p: CreateProjectParams) -> CreateProjectResponse:
        return CreateProjectResponse({
            "ok": True

        })
