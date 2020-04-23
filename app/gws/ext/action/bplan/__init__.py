"""Manage construction plans."""

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
    dataDir: t.DirPath  #: data directory
    qgisTemplate: t.FilePath  #: qgis template project


class BplanAU(t.Data):
    uid: str
    name: str


class Props(t.Props):
    type: t.Literal = 'bplan'
    auList: t.List[BplanAU]


class UploadParams(t.Params):
    uploadUid: str


class UploadResponse(t.Response):
    pass


class GetFeaturesParams(t.Params):
    auUid: str


class GetFeaturesResponse(t.Response):
    features: t.List[t.FeatureProps]


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
        self.data_dir = self.var('dataDir')
        self.qgis_template = self.var('qgisTemplate')

        for sub in 'png', 'pdf', 'vrt', 'qgs':
            gws.ensure_dir(self.data_dir + '/' + sub)

        self.key_col = 'plan_id'
        self.au_key_col = 'ags'
        self.au_name_col = 'gemeinde'
        self.type_col = 'typ'
        self.type_mapping = {
            'Flächennutzungsplan': 'F',
        }

    @gws.cached_property
    def au_list(self):
        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT DISTINCT {self.au_key_col}, {self.au_name_col} 
                FROM {conn.quote_table(self.table.name)}
                ORDER BY {self.au_name_col}
            ''')
            return [t.Data(uid=r[self.au_key_col], name=r[self.au_name_col]) for r in rs]

    def props_for(self, user):
        return {
            'type': self.type,
            'auList': self.au_list,
        }

    def api_get_features(self, req: t.IRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        features = self.db.select(t.SelectArgs(
            table=self.table,
            extra_where=[f'{self.au_key_col} = %s', p.auUid],
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

        try:
            importer.run(self, rec.path, False)
        except Exception as e:
            gws.log.exception()
            raise gws.web.error.BadRequest()

        return UploadResponse()

    def do_import(self, path, replace):
        importer.run(self, path, replace)

    def do_update(self):
        importer.update(self)
