import gws
import gws.base.action
import gws.lib.mime
import gws.lib.zipx
import gws.lib.osx
import gws.types as t

from . import core

gws.ext.new.action('qfield')


class Config(gws.ConfigWithAccess):
    """QField action."""

    packages: list[core.PackageConfig]


class Props(gws.base.action.Props):
    pass


class ExportRequest(gws.Request):
    packageUid: str
    namePrefix: str = ''
    omitBaseMap: bool = False
    omitMedia: bool = False
    storeDbInDCIM: bool = False


class ExportResponse(gws.Response):
    data: bytes


class ImportRequest(gws.Request):
    packageUid: str
    data: t.Optional[bytes]


class ImportResponse(gws.Response):
    pass


class Object(gws.base.action.Object):
    packages: dict[str, core.Package]

    def configure(self):
        self.packages = {
            p.uid: p
            for p in self.create_children(core.Package, self.cfg('packages'))
        }

    @gws.ext.command.get('qfieldExport')
    def http_export(self, req: gws.IWebRequester, p: ExportRequest) -> gws.ContentResponse:
        b = self._do_export(req, p)
        return gws.ContentResponse(content=b, mime=gws.lib.mime.ZIP)

    @gws.ext.command.api('qfieldExport')
    def api_export(self, req: gws.IWebRequester, p: ExportRequest) -> ExportResponse:
        b = self._do_export(req, p)
        return ExportResponse(data=b)

    @gws.ext.command.post('qfieldImport')
    def http_import(self, req: gws.IWebRequester, p: ImportRequest) -> gws.ContentResponse:
        self._do_import(req, p, req.data())
        return gws.ContentResponse(content='ok\n')

    @gws.ext.command.api('qfieldImport')
    def api_import(self, req: gws.IWebRequester, p: ImportRequest) -> ImportResponse:
        self._do_import(req, p, p.data)
        return ImportResponse()

    ##

    def _do_export(self, req: gws.IWebRequester, p: ExportRequest) -> bytes:
        project = req.require_project(p.projectUid)
        package = self._get_package(p.packageUid, req.user, gws.Access.read)

        base_dir = gws.ensure_dir(f'{gws.VAR_DIR}/qfield/{gws.random_string(32)}')
        opts = core.ExportOptions(
            baseDir=base_dir,
            namePrefix='',
            storeDbInDCIM=p.storeDbInDCIM,
            withData=True,
            withQgis=True,
            withBaseMap=not p.omitBaseMap,
            withMedia=not p.omitMedia,
        )

        core.export_package(package, project, req.user, opts)
        b = gws.lib.zipx.zip_to_bytes(base_dir)
        gws.lib.osx.unlink(base_dir)

        return b

    def _do_import(self, req: gws.IWebRequester, p: ImportRequest, data: bytes):

        project = req.require_project(p.projectUid)
        package = self._get_package(p.packageUid, req.user, gws.Access.read)

        base_dir = gws.ensure_dir(f'{gws.VAR_DIR}/qfield/{gws.random_string(32)}')

        if data.startswith(b'SQLite'):
            gws.write_file_b(f'{base_dir}/{package.uid}.{core.GPKG_EXT}', data)
        else:
            gws.lib.zipx.unzip_bytes(data, base_dir, flat=True)

        opts = core.ImportOptions(
            baseDir=base_dir,
        )

        core.import_data_from_package(package, project, req.user, opts)
        gws.lib.osx.unlink(base_dir)

    def _get_package(self, uid: str, user: gws.IUser, access: gws.Access) -> core.Package:
        pkg = self.packages.get(uid)
        if not pkg:
            raise gws.NotFoundError(f'package {uid} not found')
        if not user.can(access, pkg):
            raise gws.ForbiddenError(f'package {uid} forbidden')
        return pkg
