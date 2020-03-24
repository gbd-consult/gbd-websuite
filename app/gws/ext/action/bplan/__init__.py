"""Manage construction plans."""

import re
import shutil
import zipfile

import gws
import gws.common.action
import gws.common.db
import gws.common.template
import gws.tools.upload
import gws.ext.db.provider.postgres
import gws.web.error

import gws.types as t

from . import importer


class Config(t.WithTypeAndAccess):
    """Construction plans action"""

    db: str = ''  #: database provider ID
    crs: t.Crs  #: CRS for the bplan data
    table: gws.common.db.SqlTableConfig  #: sql table configuration


class BplanArea(t.Data):
    uid: str
    name: str


class Props(t.Props):
    type: t.Literal = 'bplan'
    areas: t.List[BplanArea]


class UploadParams(t.Params):
    uploadUid: str


class UploadResponse(t.Response):
    pass


class GetFeaturesParams(t.Params):
    areaCode: str


class GetFeaturesResponse(t.Response):
    features: t.List[t.FeatureProps]


_AREA_KEY_FIELD = 'ags'
_AREA_NAME_FIELD = 'gemeinde'

DEFAULT_FORMAT = gws.common.template.FeatureFormatConfig(
    title=gws.common.template.Config(type='html', text='{name}'),
    teaser=gws.common.template.Config(type='html', text='{name}'),
)


class Object(gws.common.action.Object):

    def configure(self):
        super().configure()

        self.crs = self.var('crs')
        self.db = t.cast(
            gws.ext.db.provider.postgres.Object,
            gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres'))

        self.feature_format = t.cast(t.IFormat, self.create_child('gws.common.format', DEFAULT_FORMAT))

        self.table = self.db.configure_table(self.var('table'))

        areas = {}
        with self.db.connect() as conn:
            rs = conn.select(f"SELECT {_AREA_KEY_FIELD}, {_AREA_NAME_FIELD} FROM {self.table.name}")
            for r in rs:
                areas[r[_AREA_KEY_FIELD]] = r[_AREA_NAME_FIELD]

        self.areas = sorted(areas.items(), key=lambda x: x[1])

    def props_for(self, user):
        return {
            'type': self.type,
            'areas': [t.Data(uid=a[0], name=a[1]) for a in self.areas]
        }

    def api_get_features(self, req: t.IRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        features = self.db.select(t.SelectArgs(
            table=self.table,
            extra_where=[f'{_AREA_KEY_FIELD} = %s', p.areaCode],
        ))
        return GetFeaturesResponse(features=[f.apply_format(self.feature_format).props for f in features])

    def api_upload_chunk(self, req: t.IRequest, p: gws.tools.upload.UploadChunkParams) -> gws.tools.upload.UploadChunkResponse:
        return gws.tools.upload.upload_chunk(p)

    def api_upload(self, req: t.IRequest, p: UploadParams) -> UploadResponse:
        try:
            rec = gws.tools.upload.get(p.uploadUid)
        except gws.tools.upload.Error as e:
            gws.log.error(e)
            raise gws.web.error.BadRequest()

        uid = gws.random_string(64)
        dir = gws.ensure_dir(gws.TMP_DIR + '/bplan/' + uid)

        try:
            self._extract_upload(rec.path, dir)
            importer.run(self.db, self.table, dir)
        except Exception as e:
            gws.log.exception()
            raise gws.web.error.BadRequest()

        return UploadResponse()

    def _extract_upload(self, zip_path, target_dir):
        zf = zipfile.ZipFile(zip_path)
        for fi in zf.infolist():
            fn = re.sub(r'[\\/]', '__', fi.filename)
            with zf.open(fi) as src, open(target_dir + '/' + fn, 'wb') as dst:
                shutil.copyfileobj(src, dst)
