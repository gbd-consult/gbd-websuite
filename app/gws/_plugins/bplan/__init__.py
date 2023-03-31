"""Manage construction plans."""

import gws.ext.db.provider.postgres
import gws.ext.helper.csv

import gws
import gws.types as t
import gws.base.api
import gws.base.db
import gws.lib.metadata
import gws.base.model
import gws.base.template
import gws.lib.shape
import gws.lib.date
import gws.lib.job
import gws.lib.json2
import gws.lib.upload
import gws.server.spool
import gws.base.web.error
from . import importer


class AdministrativeUnitConfig(gws.WithAccess):
    uid: str
    name: str


class PlanTypeConfig(gws.Config):
    uid: str
    name: str
    srcName: str
    color: str


class Config(gws.WithAccess):
    """Construction plans management action"""

    db: str = ''  #: database provider ID
    crs: gws.Crs  #: CRS for the bplan data
    planTable: gws.base.db.SqlTableConfig  #: plan table configuration
    metaTable: gws.base.db.SqlTableConfig  #: meta table configuration
    dataDir: gws.DirPath  #: data directory
    templates: list[gws.ext.template.Config]  #: templates
    administrativeUnits: list[AdministrativeUnitConfig]  #: Administrative Units
    planTypes: list[PlanTypeConfig]  #: Plan Types
    imageQuality: int = 24  #: palette size for optimized images
    uploadChunkSize: int  #: upload chunk size in mb
    exportDataModel: t.Optional[gws.base.model.Config]  #: data model for csv export


class AdministrativeUnit(gws.Data):
    uid: str
    name: str


class Props(gws.Props):
    type: t.Literal = 'bplan'
    auList: list[AdministrativeUnit]
    uploadChunkSize: int


class StatusParams(gws.Params):
    jobUid: str


class StatusResponse(gws.Response):
    jobUid: str
    progress: int
    state: gws.lib.job.State
    stats: importer.Stats


class ImportParams(gws.Params):
    uploadUid: str
    auUid: str
    replace: bool


class GetFeaturesParams(gws.Params):
    auUid: str


class GetFeaturesResponse(gws.Response):
    features: list[gws.lib.feature.Props]


class DeleteFeatureParams(gws.Params):
    uid: str


class DeleteFeatureResponse(gws.Response):
    pass


class LoadUserMetaParams(gws.Params):
    auUid: str


class LoadUserMetaResponse(gws.Response):
    meta: dict


class SaveUserMetaParams(gws.Params):
    auUid: str
    meta: dict


class SaveUserMetaResponse(gws.Response):
    pass


class LoadInfoParams(gws.Params):
    auUid: str


class LoadInfoResponse(gws.Response):
    info: str


class ExportParams(gws.Params):
    auUid: str


class ExportResponse(gws.Response):
    fileName: str
    content: str
    mime: str


_RELOAD_FILE = gws.VAR_DIR + '/bplan.reload'


class Object(gws.base.api.Action):

    def configure(self):
        

        self.crs = self.cfg('crs')
        self.db = t.cast(
            gws.ext.db.provider.postgres.Object,
            gws.base.db.require_provider(self, 'gws.ext.db.provider.postgres'))

        self.templates: list[gws.ITemplate] = gws.base.template.bundle(self, self.cfg('templates'))
        self.qgis_template: gws.ITemplate = gws.base.template.find(self.templates, subject='bplan.qgis')
        self.info_template: gws.ITemplate = gws.base.template.find(self.templates, subject='bplan.info')

        self.plan_table = self.db.configure_table(self.cfg('planTable'))
        self.meta_table = self.db.configure_table(self.cfg('metaTable'))
        self.data_dir = self.cfg('dataDir')

        self.au_list = self.cfg('administrativeUnits')
        self.type_list = self.cfg('planTypes')
        self.image_quality = self.cfg('imageQuality')

        p = self.cfg('exportDataModel')
        self.export_data_model: t.Optional[gws.IDataModel] = self.create_child('gws.base.model', p) if p else None

        for sub in 'png', 'pdf', 'cnv', 'qgs':
            gws.ensure_dir(self.data_dir + '/' + sub)

        self.key_col = 'plan_id'
        self.au_key_col = 'ags'
        self.au_name_col = 'gemeinde'
        self.type_col = 'typ'
        self.time_col = 'rechtskr'
        self.x_coord_col = 'utm_ost'
        self.y_coord_col = 'utm_nord'

        gws.write_file(_RELOAD_FILE, gws.random_string(16))
        self.root.application.monitor.add_path(_RELOAD_FILE)

    def post_configure(self):
        super().post_configure()
        self._load_db_meta()

    def props_for(self, user):
        return {
            'type': self.type,
            'auList': self._au_list_for(user),
            'uploadChunkSize': self.cfg('uploadChunkSize') * 1024 * 1024,
        }

    def api_get_features(self, req: gws.IWebRequest, p: GetFeaturesParams) -> GetFeaturesResponse:
        au_uid = self._check_au(req, p.auUid)

        features = self.db.select(gws.SqlSelectArgs(
            table=self.plan_table,
            extra_where=[f'_au = %s', au_uid],
            sort='name',
        ))
        for f in features:
            f.apply_templates(self.templates)
            g = f.attr('_geom_p') or f.attr('_geom_l') or f.attr('_geom_x')
            if g:
                f.shape = gws.lib.shape.from_wkb_hex(g, self.plan_table.geometry_crs)
            f.attributes = []

        return GetFeaturesResponse(features=[f.props for f in features])

    def api_delete_feature(self, req: gws.IWebRequest, p: DeleteFeatureParams) -> DeleteFeatureResponse:
        with self.db.connect() as conn:
            r = conn.select_one(f'''
                SELECT *
                FROM {conn.quote_table(self.plan_table.name)}
                WHERE _uid = %s
            ''', [p.uid])

        if not r:
            raise gws.base.web.error.NotFound()

        self._check_au(req, r['_au'])

        importer.delete_feature(self, r['_uid'])

        return DeleteFeatureResponse()

    def api_upload_chunk(self, req: gws.IWebRequest, p: gws.lib.upload.UploadChunkParams) -> gws.lib.upload.UploadChunkResponse:
        return gws.lib.upload.upload_chunk(p)

    def api_import(self, req: gws.IWebRequest, p: ImportParams) -> StatusResponse:
        try:
            rec = gws.lib.upload.get(p.uploadUid)
        except gws.lib.upload.Error as e:
            gws.log.error(e)
            raise gws.base.web.error.BadRequest()

        job_uid = gws.random_string(64)

        args = {
            'actionUid': self.uid,
            'auUid': self._check_au(req, p.auUid),
            'path': rec.path,
            'replace': p.replace,
        }

        job = gws.lib.job.create(
            uid=job_uid,
            user=req.user,
            args=gws.lib.json2.to_string(args),
            worker=__name__ + '._worker')

        gws.server.spool.add(job)

        return StatusResponse(
            jobUid=job.uid,
            state=job.state,
        )

    def api_import_status(self, req: gws.IWebRequest, p: StatusParams) -> StatusResponse:
        job = gws.lib.job.get_for(req.user, p.jobUid)
        if not job:
            raise gws.base.web.error.NotFound()

        return StatusResponse(
            jobUid=job.uid,
            state=job.state,
            progress=job.progress,
            stats=job.result.get('stats', {}) if job.result else {},
        )

    def api_import_cancel(self, req: gws.IWebRequest, p: StatusParams) -> StatusResponse:
        """Cancel a print job"""

        job = gws.lib.job.get_for(req.user, p.jobUid)
        if not job:
            raise gws.base.web.error.NotFound()

        job.cancel()

        return StatusResponse(
            jobUid=job.uid,
            state=job.state,
        )

    def api_load_user_meta(self, req: gws.IWebRequest, p: LoadUserMetaParams) -> LoadUserMetaResponse:
        """Return the user metadata"""

        au_uid = self._check_au(req, p.auUid)

        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT * 
                FROM {conn.quote_table(self.meta_table.name)} 
                WHERE _au=%s
            ''', [au_uid])
            for r in rs:
                return LoadUserMetaResponse(meta=gws.lib.json2.from_string(r['meta']))

        return LoadUserMetaResponse(meta={})

    def api_save_user_meta(self, req: gws.IWebRequest, p: SaveUserMetaParams) -> SaveUserMetaResponse:

        au_uid = self._check_au(req, p.auUid)

        with self.db.connect() as conn:
            with conn.transaction():
                conn.execute(f'''
                    DELETE
                    FROM {conn.quote_table(self.meta_table.name)}
                    WHERE _au=%s
                ''', [au_uid])

                conn.execute(f'''
                    INSERT 
                    INTO {conn.quote_table(self.meta_table.name)}
                    (_au, user_id, meta)
                    VALUES(%s, %s, %s)
                ''', [au_uid, req.user.fid, gws.lib.json2.to_pretty_string(p.meta)])

        self._load_db_meta()
        self.signal_reload('metadata')

        return SaveUserMetaResponse()

    def api_load_info(self, req: gws.IWebRequest, p: LoadInfoParams) -> LoadInfoResponse:
        res = self.info_template.render({'auUid': p.auUid})
        return LoadInfoResponse(info=res.content)

    def api_csv_export(self, req: gws.IWebRequest, p: ExportParams) -> ExportResponse:
        au_uid = self._check_au(req, p.auUid)

        features = self.db.select(gws.SqlSelectArgs(
            table=self.plan_table,
            extra_where=[f'_au = %s', au_uid],
            sort='name',
        ))

        helper: gws.ext.helper.csv.Object = t.cast(
            gws.ext.helper.csv.Object,
            self.root.find_first('gws.ext.helper.csv'))
        writer = helper.writer()
        has_headers = False

        for f in features:
            if self.export_data_model:
                f.apply_data_model(self.export_data_model)
            if not has_headers:
                writer.write_headers([a.name for a in f.attributes])
                has_headers = True
            writer.write_attributes(f.attributes)

        return ExportResponse(
            fileName='bauplaene_' + au_uid + '.csv',
            content=writer.as_bytes(),
            mime='text/csv'
        )

    def do_import(self, path, replace):
        importer.run(self, path, replace)

    def do_update(self):
        importer.update(self)

    def signal_reload(self, source):
        gws.log.debug(f'bplan reload signal {source!r}')
        gws.write_file(_RELOAD_FILE, gws.random_string(16))

    def _au_list_for(self, user):
        return [
            gws.Data(uid=au.uid, name=au.name)
            for au in self.au_list
            if user.can_use(au)
        ]

    def _check_au(self, req, au_uid):
        au_uids = set(au.uid for au in self._au_list_for(req.user))

        if au_uid not in au_uids:
            gws.log.error(f'wrong auUid={au_uid}')
            raise gws.base.web.error.Forbidden()

        return au_uid

    def _load_db_meta(self):
        metas = {}
        meta_dates = {}

        def _date(s):
            return gws.lib.date.to_iso(gws.lib.date.to_utc(s), with_tz='Z')

        with self.db.connect() as conn:
            rs = conn.select(f'''SELECT * FROM {conn.quote_table(self.meta_table.name)}''')
            for r in rs:
                au_uid = r['_au']
                metas[au_uid] = gws.lib.metadata.from_dict(gws.lib.json2.from_string(r['meta']))
                meta_dates[au_uid] = _date(r['_updated'])

            rs = conn.select(f'''
                SELECT _au, 
                    MIN(_updated) AS min_updated, 
                    MAX(_updated) AS max_updated,
                    MIN(CASE WHEN {self.time_col} != '' THEN DATE({self.time_col}) ELSE _updated END) AS min_time,
                    MAX(CASE WHEN {self.time_col} != '' THEN DATE({self.time_col}) ELSE _updated END) AS max_time
                FROM {conn.quote_table(self.plan_table.name)}
                GROUP BY _au
            ''')

            for r in rs:
                au_uid = r['_au']
                if au_uid not in metas:
                    metas[au_uid] = gws.Values()

                metas[au_uid].dateCreated = _date(r['min_updated'])
                metas[au_uid].dateUpdated = _date(r['max_updated'])
                md = meta_dates.get(au_uid)
                if md and md < metas[au_uid].dateCreated:
                    metas[au_uid].dateCreated = md
                if md and md > metas[au_uid].dateUpdated:
                    metas[au_uid].dateUpdated = md

                metas[au_uid].dateBegin = _date(r['min_time'])
                metas[au_uid].dateEnd = _date(r['max_time'])

        # extend metadata for "our" objects

        for obj in self.root.find_all():
            uid = gws.get(obj, 'uid') or ''
            if uid and gws.get(obj, 'meta'):
                for au_uid, meta in metas.items():
                    if uid.endswith(au_uid):
                        obj.meta = gws.lib.metadata.extend(meta, obj.meta)
                        if gws.get(obj, 'update_sequence'):
                            obj.update_sequence = meta.dateUpdated


def _worker(root: gws.RootObject, job: gws.lib.job.Job):
    args = gws.lib.json2.from_string(job.args)
    action = t.cast(Object, root.find('gws.ext.action', args['actionUid']))
    job.update(state=gws.lib.job.State.running)
    stats = importer.run(action, args['path'], args['replace'], args['auUid'], job)
    job.update(result={'stats': stats})
    action.signal_reload('worker')
