from typing import Optional, cast

import gws
import gws.config
import gws.base.action
import gws.base.database
import gws.lib.mime
import gws.lib.date
import gws.lib.zipx
import gws.lib.osx
import gws.lib.sa as sa

from . import core

gws.ext.new.action('qfield')


class Config(gws.ConfigWithAccess):
    """QField action."""

    packages: list[core.PackageConfig]
    withDbInDCIM: bool = False
    withSerialPrefix: bool = False


class Props(gws.base.action.Props):
    pass


class DownloadRequest(gws.Request):
    packageUid: Optional[str]
    omitStatic: bool = False
    omitData: bool = False


class PackageParams(gws.CliParams):
    projectUid: str
    packageUid: str
    out: str


class DownloadResponse(gws.Response):
    data: bytes


class UploadRequest(gws.Request):
    packageUid: Optional[str]
    data: Optional[bytes]


class UploadResponse(gws.Response):
    pass


_SERIAL_PREFIX_END_DATE = '2033-01-01'


class Object(gws.base.action.Object):
    packages: dict[str, core.Package]
    withDbInDCIM: bool
    withSerialPrefix: bool

    def configure(self):
        self.packages = {
            p.uid: p
            for p in self.create_children(core.Package, self.cfg('packages'))
        }
        self.withDbInDCIM = self.cfg('withDbInDCIM', default=False)
        self.withSerialPrefix = self.cfg('withSerialPrefix', default=False)

    @gws.ext.command.get('qfieldDownload')
    def http_download(self, req: gws.WebRequester, p: DownloadRequest) -> gws.ContentResponse:
        b = self._do_download(req, p)
        return gws.ContentResponse(content=b, mime=gws.lib.mime.ZIP)

    @gws.ext.command.api('qfieldDownload')
    def api_download(self, req: gws.WebRequester, p: DownloadRequest) -> DownloadResponse:
        b = self._do_download(req, p)
        return DownloadResponse(data=b)

    @gws.ext.command.cli('qfieldPackage')
    def cli_package(self, p: PackageParams):
        root = gws.config.load()
        user = root.app.authMgr.systemUser
        project = user.require_project(p.projectUid)
        action = cast(Object, root.app.actionMgr.find_action(project, self.extType, user))
        args = action.prepare_export(DownloadRequest(
            projectUid=p.projectUid,
            packageUid=p.packageUid,
        ), user)
        action.exec_export(args)
        b = action.end_export(args)
        gws.u.write_file_b(p.out, b)

    @gws.ext.command.post('qfieldUpload')
    def http_upload(self, req: gws.WebRequester, p: UploadRequest) -> gws.ContentResponse:
        self._do_upload(req, p, req.data())
        return gws.ContentResponse(content='ok\n')

    @gws.ext.command.api('qfieldUpload')
    def api_upload(self, req: gws.WebRequester, p: UploadRequest) -> UploadResponse:
        self._do_upload(req, p, p.data)
        return UploadResponse()

    ##

    def _do_download(self, req: gws.WebRequester, p: DownloadRequest) -> bytes:
        args = self.prepare_export(p, req.user)
        self.exec_export(args)
        return self.end_export(args)

    def prepare_export(self, p: DownloadRequest, user: gws.User) -> core.ExportArgs:
        project = user.require_project(p.projectUid)
        package = self._get_package(p.packageUid, user, gws.Access.read)
        base_dir = gws.u.ensure_dir(f'{gws.c.VAR_DIR}/qfield/{gws.u.random_string(32)}')

        name_prefix = ''
        if self.withSerialPrefix:
            minutes = (gws.lib.date.to_timestamp(gws.lib.date.from_iso(_SERIAL_PREFIX_END_DATE)) - gws.lib.date.timestamp()) // 60
            name_prefix = '{:06d}'.format(minutes) + '_' + gws.lib.date.now().strftime('%d.%m.%y') + '_'

        db_file_name = f'{name_prefix}{package.uid}.{core.GPKG_EXT}'
        db_path = db_file_name
        if self.withDbInDCIM:
            db_path = f'DCIM/{db_file_name}'
            gws.u.ensure_dir(f'{base_dir}/DCIM')

        return core.ExportArgs(
            package=package,
            project=project,
            user=user,
            baseDir=base_dir,
            qgisFileName=f'{name_prefix}{package.uid}',
            dbFileName=db_file_name,
            dbPath=db_path,
            withBaseMap=not p.omitStatic,
            withData=not p.omitData,
            withMedia=not p.omitStatic,
            withQgis=not p.omitData,
        )

    def exec_export(self, args: core.ExportArgs):
        core.Exporter().run(args)

    def end_export(self, args: core.ExportArgs) -> bytes:
        b = gws.lib.zipx.zip_to_bytes(args.baseDir)

        if not self.root.app.developer_option('qfield.keep_temp_dirs'):
            gws.lib.osx.rmdir(args.baseDir)

        return b

    ##

    def _do_upload(self, req: gws.WebRequester, p: UploadRequest, data: bytes):
        args = self.prepare_import(p, req.user, data)
        self.exec_import(args)
        return self.end_import(args)

    def prepare_import(self, p: UploadRequest, user: gws.User, data: bytes):
        project = user.require_project(p.projectUid)
        package = self._get_package(p.packageUid, user, gws.Access.write)
        base_dir = gws.u.ensure_dir(f'{gws.c.VAR_DIR}/qfield/{gws.u.random_string(32)}')

        if data.startswith(b'SQLite'):
            gws.u.write_file_b(f'{base_dir}/{package.uid}.{core.GPKG_EXT}', data)
        else:
            gws.lib.zipx.unzip_bytes(data, base_dir, flat=True)

        db_file_name = f'{package.uid}.{core.GPKG_EXT}'

        return core.ImportArgs(
            package=package,
            project=project,
            user=user,
            baseDir=base_dir,
            dbFileName=db_file_name,
        )

    def exec_import(self, args: core.ImportArgs):
        core.Importer().run(args)

    def end_import(self, args: core.ImportArgs):
        if not self.root.app.developer_option('qfield.keep_temp_dirs'):
            gws.lib.osx.rmdir(args.baseDir)

    ##

    def _get_package(self, uid: str, user: gws.User, access: gws.Access) -> core.Package:
        pkg = self.packages.get(uid) if uid else gws.u.first(self.packages.values())
        if not pkg:
            raise gws.NotFoundError(f'package {uid} not found')
        if not user.can(access, pkg):
            raise gws.ForbiddenError(f'package {uid} forbidden')
        return pkg
