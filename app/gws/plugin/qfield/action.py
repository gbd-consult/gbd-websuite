import gws
import gws.base.action
import gws.base.database
import gws.lib.mime
import gws.lib.date
import gws.lib.zipx
import gws.lib.osx
import gws.lib.sa as sa
import gws.types as t

from . import core

gws.ext.new.action('qfield')


class Config(gws.ConfigWithAccess):
    """QField action."""

    packages: list[core.PackageConfig]
    defaultProjectUid: str = ''
    defaultPackageUid: str = ''
    withDbInDCIM: bool = False
    withSerialPrefix: bool = False


class Props(gws.base.action.Props):
    pass


class DownloadRequest(gws.Request):
    packageUid: t.Optional[str]
    omitStatic: bool = False
    omitData: bool = False


class DownloadResponse(gws.Response):
    data: bytes


class UploadRequest(gws.Request):
    packageUid: t.Optional[str]
    data: t.Optional[bytes]


class UploadResponse(gws.Response):
    pass


_SERIAL_PREFIX_END_DATE = '2033-01-01'


class Object(gws.base.action.Object):
    packages: dict[str, core.Package]
    defaultProjectUid: str
    defaultPackageUid: str
    withDbInDCIM: bool
    withSerialPrefix: bool

    def configure(self):
        self.packages = {
            p.uid: p
            for p in self.create_children(core.Package, self.cfg('packages'))
        }
        self.defaultProjectUid = self.cfg('defaultProjectUid', default='')
        self.defaultPackageUid = self.cfg('defaultPackageUid', default='')
        self.withDbInDCIM = self.cfg('withDbInDCIM', default=False)
        self.withSerialPrefix = self.cfg('withSerialPrefix', default=False)

    @gws.ext.command.get('qfieldDownload')
    def http_download(self, req: gws.IWebRequester, p: DownloadRequest) -> gws.ContentResponse:
        b = self._do_download(req, p)
        return gws.ContentResponse(content=b, mime=gws.lib.mime.ZIP)

    @gws.ext.command.api('qfieldDownload')
    def api_download(self, req: gws.IWebRequester, p: DownloadRequest) -> DownloadResponse:
        b = self._do_download(req, p)
        return DownloadResponse(data=b)

    @gws.ext.command.post('qfieldUpload')
    def http_upload(self, req: gws.IWebRequester, p: UploadRequest) -> gws.ContentResponse:
        self._do_upload(req, p, req.data())
        return gws.ContentResponse(content='ok\n')

    @gws.ext.command.api('qfieldUpload')
    def api_upload(self, req: gws.IWebRequester, p: UploadRequest) -> UploadResponse:
        self._do_upload(req, p, p.data)
        return UploadResponse()

    ##

    def _do_download(self, req: gws.IWebRequester, p: DownloadRequest) -> bytes:
        project = req.require_project(p.projectUid or self.defaultProjectUid)
        package = self._get_package(
            p.packageUid or self.defaultPackageUid,
            req.user,
            gws.Access.read
        )

        name_prefix = ''
        if self.withSerialPrefix:
            mins = (gws.lib.date.to_timestamp(gws.lib.date.from_iso(_SERIAL_PREFIX_END_DATE)) - gws.lib.date.timestamp()) // 60
            name_prefix = '{:06d}'.format(mins) + '_' + gws.lib.date.now().strftime('%d.%m.%y') + '_'

        base_dir = gws.ensure_dir(f'{gws.VAR_DIR}/qfield/{gws.random_string(32)}')

        opts = core.ExportOptions(
            baseDir=base_dir,
            qgisFileName=f'{name_prefix}{package.uid}',
            dbFileName=f'{name_prefix}{package.uid}',
            withDbInDCIM=self.withDbInDCIM,
            withData=not p.omitData,
            withQgis=not p.omitData,
            withBaseMap=not p.omitStatic,
            withMedia=not p.omitStatic,
        )

        core.export_package(package, project, req.user, opts)
        b = gws.lib.zipx.zip_to_bytes(base_dir)

        if not self.root.app.developer_option('qfield.keep_temp_dirs'):
            gws.lib.osx.unlink(base_dir)

        return b

    def _do_upload(self, req: gws.IWebRequester, p: UploadRequest, data: bytes):

        project = req.require_project(p.projectUid or self.defaultProjectUid)
        package = self._get_package(
            p.packageUid or self.defaultPackageUid,
            req.user,
            gws.Access.write
        )

        base_dir = gws.ensure_dir(f'{gws.VAR_DIR}/qfield/{gws.random_string(32)}')

        if data.startswith(b'SQLite'):
            gws.write_file_b(f'{base_dir}/{package.uid}.{core.GPKG_EXT}', data)
        else:
            gws.lib.zipx.unzip_bytes(data, base_dir, flat=True)

        opts = core.ImportOptions(
            baseDir=base_dir,
        )

        core.import_data_from_package(package, project, req.user, opts)

        if not self.root.app.developer_option('qfield.keep_temp_dirs'):
            gws.lib.osx.unlink(base_dir)

    def _get_package(self, uid: str, user: gws.IUser, access: gws.Access) -> core.Package:
        pkg = self.packages.get(uid)
        if not pkg:
            raise gws.NotFoundError(f'package {uid} not found')
        if not user.can(access, pkg):
            raise gws.ForbiddenError(f'package {uid} forbidden')
        return pkg
