"""Manage construction plans."""

import gws
import gws.common.action
import gws.common.db
import gws.common.template
import gws.ext.db.provider.postgres
import gws.server.spool
import gws.tools.job
import gws.tools.json2
import gws.tools.upload
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


class ImportParams(t.Params):
    uploadUid: str
    replace: bool


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

    def api_import(self, req: t.IRequest, p: ImportParams) -> gws.tools.job.StatusResponse:
        try:
            rec = gws.tools.upload.get(p.uploadUid)
        except gws.tools.upload.Error as e:
            gws.log.error(e)
            raise gws.web.error.BadRequest()

        job_uid = gws.random_string(64)

        args = {
            'actionUid': self.uid,
            'path': rec.path,
            'replace': p.replace,
        }

        job = gws.tools.job.create(
            uid=job_uid,
            user=req.user,
            args=gws.tools.json2.to_string(args),
            worker=__name__ + '._worker')

        gws.server.spool.add(job)

        return gws.tools.job.StatusResponse(
            jobUid=job.uid,
            state=job.state,
        )

    def api_import_status(self, req: t.IRequest, p: gws.tools.job.StatusParams) -> gws.tools.job.StatusResponse:
        r = gws.tools.job.status_request(req, p)
        if not r:
            raise gws.web.error.NotFound()
        return r
        #
        #
        #
        #
        # return UploadResponse()

    def api_import_cancel(self, req: t.IRequest, p: gws.tools.job.StatusParams) -> gws.tools.job.StatusResponse:
        """Cancel a print job"""

        r = gws.tools.job.cancel_request(req, p)
        if not r:
            raise gws.web.error.NotFound()
        return r

    def do_import(self, path, replace):
        importer.run(self, path, replace)

    def do_update(self):
        importer.update(self)


def _worker(root: t.IRootObject, job: gws.tools.job.Job):
    args = gws.tools.json2.from_string(job.args)
    action = root.find('gws.ext.action', args['actionUid'])
    job.update(state=gws.tools.job.State.running)
    importer.run(action, job, args['path'], args['replace'])
