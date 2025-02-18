"""Provides the printing API."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.config
import gws.lib.jsonx
import gws.lib.mime
import gws.lib.style

gws.ext.new.action('printer')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class CliParams(gws.CliParams):
    project: Optional[str]
    """project uid"""
    request: str
    """path to request.json"""
    output: str
    """output path"""


class Object(gws.base.action.Object):

    @gws.ext.command.api('printerStart')
    def printer_start(self, req: gws.WebRequester, p: gws.PrintRequest) -> gws.JobResponse:
        """Start a background print job"""

        mgr = self.root.app.printerMgr
        return mgr.start_print_job(p, req.user)

    @gws.ext.command.cli('printerPrint')
    def print(self, p: CliParams):
        """Print using the specified params"""

        root = gws.config.load()
        request = root.specs.read(
            gws.lib.jsonx.from_path(p.request),
            'gws.PrintRequest',
            p.request,
        )

        mgr = root.app.printerMgr
        res_path = mgr.exec_print(request, root.app.authMgr.systemUser)
        gws.u.write_file_b(p.output, gws.u.read_file_b(res_path))
