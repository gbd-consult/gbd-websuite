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


class AdministrativeUnitConfig(t.WithAccess):
    uid: str
    name: str


class Config(t.WithTypeAndAccess):
    """Construction plans action"""

    db: str = ''  #: database provider ID
    crs: t.Crs  #: CRS for the bplan data
    planTable: gws.common.db.SqlTableConfig  #: plan table configuration
    metaTable: gws.common.db.SqlTableConfig  #: meta table configuration
    dataDir: t.DirPath  #: data directory
    qgisTemplate: t.FilePath  #: qgis template project
    administrativeUnits: t.List[AdministrativeUnitConfig]


class AdministrativeUnit(t.Data):
    uid: str
    name: str


class Props(t.Props):
    type: t.Literal = 'bplan'
    auList: t.List[AdministrativeUnit]


class StatusParams(t.Params):
    jobUid: str


class StatusResponse(t.Response):
    jobUid: str
    progress: int
    state: gws.tools.job.State
    stats: importer.Stats


class ImportParams(t.Params):
    uploadUid: str
    auUid: str
    replace: bool


class GetFeaturesParams(t.Params):
    auUid: str


class GetFeaturesResponse(t.Response):
    features: t.List[t.FeatureProps]


class LoadUserMetaParams(t.Params):
    pass


class LoadUserMetaResponse(t.Response):
    meta: dict


class SaveUserMetaParams(t.Params):
    meta: dict


class SaveUserMetaResponse(t.Response):
    pass


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
        self.plan_table = self.db.configure_table(self.var('planTable'))
        self.meta_table = self.db.configure_table(self.var('metaTable'))
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

        self.au_list = self.var('administrativeUnits')

    def au_list_for(self, user):
        return [au for au in self.au_list if user.can_use(au)]

    def props_for(self, user):
        return {
            'type': self.type,
            'auList': self.au_list_for(user),
        }

    def api_get_features(self, req: t.IRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        features = self.db.select(t.SelectArgs(
            table=self.plan_table,
            extra_where=[f'{self.au_key_col} = %s', p.auUid],
        ))
        return GetFeaturesResponse(features=[f.apply_format(self.feature_format).props for f in features])

    def api_upload_chunk(self, req: t.IRequest, p: gws.tools.upload.UploadChunkParams) -> gws.tools.upload.UploadChunkResponse:
        return gws.tools.upload.upload_chunk(p)

    def api_import(self, req: t.IRequest, p: ImportParams) -> StatusResponse:
        try:
            rec = gws.tools.upload.get(p.uploadUid)
        except gws.tools.upload.Error as e:
            gws.log.error(e)
            raise gws.web.error.BadRequest()

        au_uids = set(au.uid for au in self.au_list_for(req.user))

        if p.auUid not in au_uids:
            gws.log.error(f'wrong auUid={p.auUid}')
            raise gws.web.error.Forbidden()

        job_uid = gws.random_string(64)

        args = {
            'actionUid': self.uid,
            'auUid': p.auUid,
            'path': rec.path,
            'replace': p.replace,
        }

        job = gws.tools.job.create(
            uid=job_uid,
            user=req.user,
            args=gws.tools.json2.to_string(args),
            worker=__name__ + '._worker')

        gws.server.spool.add(job)

        return StatusResponse(
            jobUid=job.uid,
            state=job.state,
        )

    def api_import_status(self, req: t.IRequest, p: StatusParams) -> StatusResponse:
        job = gws.tools.job.get_for(req.user, p.jobUid)
        if not job:
            raise gws.web.error.NotFound()

        return StatusResponse(
            jobUid=job.uid,
            state=job.state,
            progress=job.progress,
            stats=job.result.get('stats', {}) if job.result else {},
        )

    def api_import_cancel(self, req: t.IRequest, p: StatusParams) -> StatusResponse:
        """Cancel a print job"""

        job = gws.tools.job.get_for(req.user, p.jobUid)
        if not job:
            raise gws.web.error.NotFound()

        job.cancel()

        return StatusResponse(
            jobUid=job.uid,
            state=job.state,
        )

    def api_load_user_meta(self, req: t.IRequest, p: LoadUserMetaParams) -> LoadUserMetaResponse:
        """Return the user metadata"""

        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT * 
                FROM {conn.quote_table(self.meta_table.name)} 
                WHERE user_id=%s
            ''', [req.user.fid])
            for r in rs:
                return LoadUserMetaResponse(meta=gws.tools.json2.from_string(r['meta']))

        return LoadUserMetaResponse(meta={})

    def api_save_user_meta(self, req: t.IRequest, p: SaveUserMetaParams) -> SaveUserMetaResponse:

        with self.db.connect() as conn:
            with conn.transaction():
                conn.execute(f'''
                    DELETE
                    FROM {conn.quote_table(self.meta_table.name)}
                    WHERE user_id=%s
                ''', [req.user.fid])

                conn.execute(f'''
                    INSERT 
                    INTO {conn.quote_table(self.meta_table.name)}
                    (user_id, meta)
                    VALUES(%s, %s)
                ''', [req.user.fid, gws.tools.json2.to_pretty_string(p.meta)])

        return SaveUserMetaResponse()

    def do_import(self, path, replace):
        importer.run(self, path, replace)

    def do_update(self):
        importer.update(self)


def _worker(root: t.IRootObject, job: gws.tools.job.Job):
    args = gws.tools.json2.from_string(job.args)
    action = root.find('gws.ext.action', args['actionUid'])
    job.update(state=gws.tools.job.State.running)
    stats = importer.run(action, args['path'], args['replace'], args['auUid'], job)
    job.update(result={'stats': stats})
