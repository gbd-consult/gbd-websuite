import gws
import gws.tools.storage

import gws.types as t


class FileWriteParams(t.Data):
    projectUid: str
    fileName: str
    model: dict


class FileWriteResponse(t.Response):
    pass


class FileReadParams(t.Data):
    projectUid: str
    fileName: str


class FileReadResponse(t.Response):
    fileName: str
    model: dict


class FileListParams(t.Data):
    projectUid: str


class FileListResponse(t.Response):
    fileNames: t.List[str]


class OptionsParams(t.Data):
    projectUid: str


class OptionsResponse(t.Response):
    layerUids: t.Optional[t.List[str]]
    pixelTolerance: int = 10


class Config(t.WithTypeAndAccess):
    """Dimension action"""
    layers: t.Optional[t.List[str]]  #: target layer uids
    pixelTolerance: int = 10  #: pixel tolerance


_FILE_DIRECTORY = 'dimension'


class Object(gws.Object):

    def api_options(self, req, p: OptionsParams) -> OptionsResponse:
        req.require_project(p.projectUid)

        return OptionsResponse({
            'layerUids': self.var('layers') or [],
            'pixelTolerance': self.var('pixelTolerance'),

        })

    def api_file_write(self, req, p: FileWriteParams) -> FileWriteResponse:
        req.require_project(p.projectUid)

        gws.tools.storage.put(_FILE_DIRECTORY, p.fileName, req.user.full_uid, p.model)

        return FileWriteResponse()

    def api_file_read(self, req, p: FileReadParams) -> FileReadResponse:
        req.require_project(p.projectUid)

        model = gws.tools.storage.get(_FILE_DIRECTORY, p.fileName, req.user.full_uid)

        return FileReadResponse({
            'fileName': p.fileName,
            'model': model
        })

    def api_file_list(self, req, p: FileListParams) -> FileListResponse:
        req.require_project(p.projectUid)

        names = gws.tools.storage.get_names(_FILE_DIRECTORY, req.user.full_uid)

        return FileListResponse({
            'fileNames': names
        })
