"""Provide configuration for the client Select module."""

import gws
import gws.common.action

import gws.types as t


class ExportFormat(t.Data):
    uid: str
    title: str
    helper: str  #: helper uid


class ExportFormatProps(t.Props):
    title: str
    uid: str


class Props(t.Props):
    type: t.Literal = 'select'
    exportFormats: t.List[ExportFormatProps]


class Config(t.WithTypeAndAccess):
    """Select action"""

    exportFormats: t.Optional[t.List[ExportFormat]]


class ExportParams(t.Params):
    exportFormatUid: str
    featureUids: t.List[str]


class Object(gws.common.action.Object):
    export_formats: t.List[ExportFormat]

    def configure(self):
        super().configure()
        self.export_formats = self.var('exportFormats') or []

    def api_export(self, req: t.IRequest, p: ExportParams) -> t.FileResponse:
        for ex in self.export_formats:
            if ex.uid == p.exportFormatUid:
                helper = self.root.find_by_uid(ex.helper)
                return helper.do_export(req, p)
        raise gws.web.error.NotFound()

    @property
    def props(self):
        return Props(
            exportFormats=[ExportFormatProps(title=f.title, uid=f.uid) for f in self.export_formats],
        )
