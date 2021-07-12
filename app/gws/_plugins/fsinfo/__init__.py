"""FS-Info."""

import io
import re
import zipfile

import gws.ext.db.provider.postgres
import gws.ext.helper.alkis

import gws
import gws.types as t
import gws.base.api
import gws.base.db
import gws.base.template
import gws.lib.proj
import gws.lib.shape
import gws.lib.os2
import gws.base.web.error


class Config(gws.WithAccess):
    """FSInfo action"""

    db: t.Optional[str]  #: database provider uid
    dataTable: gws.base.db.SqlTableConfig
    documentTable: gws.base.db.SqlTableConfig
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: client templates


class FindFlurstueckParams(gws.Params):
    gemarkung: str = ''
    flur: str = ''
    flurstueck: str = ''
    nachname: str = ''
    vorname: str = ''
    pn: str = ''


class FindFlurstueckResponse(gws.Response):
    features: t.List[gws.lib.feature.Props]
    total: int


class GetGemarkungenResponse(gws.Response):
    names: t.List[str]


class DocumentProps(gws.Props):
    uid: str
    personUid: str
    title: str
    size: int
    filename: str
    created: str


class PersonProps(gws.Props):
    uid: str
    title: str
    description: str
    documents: t.List[DocumentProps]


class GetDetailsParams(gws.Params):
    fsUid: str


class GetDetailsResponse(gws.Response):
    feature: gws.lib.feature.Props
    persons: t.List[PersonProps]


class UploadFile(gws.Data):
    data: bytes
    mimeType: str
    title: str
    filename: str


class CheckUploadParams(gws.Params):
    personUid: str
    names: t.List[str]


class CheckUploadResponse(gws.Params):
    existingNames: t.List[str]


class UploadParams(gws.Params):
    personUid: str
    files: t.List[UploadFile]


class UploadResponse(gws.Response):
    documentUids: t.List[str]


class GetDocumentParams(gws.Params):
    documentUid: str


class DeleteDocumentParams(gws.Params):
    documentUid: str


class DeleteDocumentResponse(gws.Params):
    documentUid: str


class GetDocumentsResponse(gws.Response):
    persons: t.List[PersonProps]


class DownloadParams(gws.Params):
    personUid: str


class Object(gws.base.api.Action):
    @property
    def props(self):
        return gws.Props(enabled=True)

    def configure(self):
        

        self.db = t.cast(gws.ext.db.provider.postgres.Object, gws.base.db.require_provider(self, 'gws.ext.db.provider.postgres'))
        self.data_table = self.db.configure_table(self.var('dataTable'))
        self.document_table = self.db.configure_table(self.var('documentTable'))
        self.templates: t.List[gws.ITemplate] = gws.base.template.bundle(self, self.var('templates'))
        self.details_template: gws.ITemplate = gws.base.template.find(self.templates, subject='fsinfo.details')
        self.title_template: gws.ITemplate = gws.base.template.find(self.templates, subject='fsinfo.title')

    def api_find_flurstueck(self, req: gws.IWebRequest, p: FindFlurstueckParams) -> FindFlurstueckResponse:
        """Perform a Flurstueck search"""

        conds = []
        params = []

        for key in vars(FindFlurstueckParams):
            v = p.get(key)
            if v and v.strip():
                conds.append(f'{key}=%s')
                params.append(v)

        if not conds:
            return FindFlurstueckResponse(
                features=[],
                total=0)

        fs = self.db.select(gws.SqlSelectArgs(
            table=self.data_table,
            extra_where=[' AND '.join(conds)] + params
        ))

        features = {}
        for f in fs:
            if f.uid not in features:
                props = f.apply_templates(self.templates).props
                del props.attributes
                features[f.uid] = props

        return FindFlurstueckResponse(
            features=list(features.values()),
            total=len(features))

    def api_get_gemarkungen(self, req: gws.IWebRequest, p: gws.Params) -> GetGemarkungenResponse:
        """Return a list of Gemarkung names"""

        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT 
                    DISTINCT gemarkung
                FROM 
                    {self.data_table.name}
                ORDER BY 
                    gemarkung
            ''')

            return GetGemarkungenResponse(names=[r['gemarkung'] for r in rs])

    def api_get_details(self, req: gws.IWebRequest, p: GetDetailsParams) -> GetDetailsResponse:
        fs_key = self.data_table.key_column

        fs = self.db.select(gws.SqlSelectArgs(
            table=self.data_table,
            extra_where=[f'{fs_key} = %s', p.fsUid]
        ))
        if not fs:
            raise gws.base.web.error.NotFound()

        feature = None
        persons = []
        docs = self._all_documents()

        for f in fs:
            if not feature:
                feature = f
            data = f.attr_dict

            if data['pn']:
                persons.append(PersonProps(
                    uid=str(data['pn']),
                    title=self.title_template.render({
                        'feature': feature,
                        'data': data,
                    }).content,
                    description=self.details_template.render({
                        'feature': feature,
                        'data': data,
                    }).content,
                    documents=[d for d in docs if d.personUid == str(data['pn'])]
                ))

        fprops = feature.apply_templates(self.templates).props
        del fprops.attributes

        return GetDetailsResponse(feature=fprops, persons=persons)

    def api_check_upload(self, req: gws.IWebRequest, p: CheckUploadParams) -> CheckUploadResponse:
        if not p.names:
            return CheckUploadResponse(existingNames=[])

        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT 
                    filename
                FROM
                    {self.document_table.name}
                WHERE
                    filename {_IN(p.names)}
                    AND deleted=0
            ''', p.names)

            return CheckUploadResponse(existingNames=[r['filename'] for r in rs])

    def api_upload(self, req: gws.IWebRequest, p: UploadParams) -> UploadResponse:
        uids = self._do_upload(p, req.user.uid)
        return UploadResponse(documentUids=uids)

    def api_delete_document(self, req: gws.IWebRequest, p: DeleteDocumentParams) -> DeleteDocumentResponse:
        uid = p.documentUid

        with self.db.connect() as conn:
            conn.exec(f'''
                UPDATE {self.document_table.name}
                SET deleted=1
                WHERE {self.document_table.key_column} = %s
            ''', [uid])

            return DeleteDocumentResponse(documentUid=str(uid))

    def api_get_documents(self, req: gws.IWebRequest, p: gws.Params) -> GetDocumentsResponse:
        docs = self._all_documents()

        if not docs:
            return GetDocumentsResponse(persons=[])

        person_uids = list(set(d.personUid for d in docs))
        persons = []

        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT 
                    *
                FROM
                    {self.data_table.name}
                WHERE 
                    pn {_IN(person_uids)}
                ORDER BY
                    nachname, vorname
            ''', person_uids)

            uids = set()

            for data in rs:
                uid = str(data['pn'])

                if uid in uids:
                    continue
                uids.add(uid)

                persons.append(PersonProps(
                    uid=uid,
                    title=self.title_template.render({
                        'data': data,
                    }).content,
                    description=self.details_template.render({
                        'data': data,
                    }).content,
                    documents=[d for d in docs if d.personUid == uid]
                ))

        return GetDocumentsResponse(persons=persons)

    def http_get_document(self, req: gws.IWebRequest, p: GetDocumentParams) -> gws.ContentResponse:
        with self.db.connect() as conn:
            doc = conn.select_one(f'''
                SELECT 
                    * 
                FROM 
                    {self.document_table.name} 
                WHERE 
                    {self.document_table.key_column} = %s
                    AND deleted=0
            ''', [p.documentUid])

        if not doc:
            raise gws.base.web.error.NotFound()

        return gws.ContentResponse(
            mime=doc['mimetype'],
            content=doc['data']
        )

    def http_get_download(self, req: gws.IWebRequest, p: DownloadParams) -> gws.ContentResponse:
        with self.db.connect() as conn:
            docs = list(conn.select(f'''
                SELECT
                    filename, data
                FROM
                    {self.document_table.name}
                WHERE
                    pn = %s
                    AND deleted=0
                ORDER BY
                    title
            ''', [p.personUid]))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for d in docs:
                zf.writestr(d['filename'], d['data'])

        return gws.ContentResponse(
            mime='application/zip',
            content=buf.getvalue(),
        )

    def do_read(self, base_dir):
        by_pn = {}

        for path in gws.lib.os2.find_files(base_dir, ext='pdf'):
            rp = gws.lib.os2.rel_path(path, base_dir)
            m = re.match(r'^(\d+)/', rp)
            if not m:
                gws.log.warn(f'{path!r} - no pn found')
                continue
            pn = m.group(1)
            if pn not in by_pn:
                by_pn[pn] = []
            by_pn[pn].append(path)

        for pn, paths in by_pn.items():
            params = UploadParams(
                personUid=pn,
                files=[]
            )
            for path in paths:
                params.files.append(UploadFile(
                    data=gws.read_file_b(path),
                    mimeType='application/pdf',
                    title='',
                    filename=gws.lib.os2.parse_path(path)['filename'],
                ))
            self._do_upload(params, 'script')

    ##

    def _do_upload(self, p: UploadParams, user_uid):

        names = [f.filename for f in p.files]
        uids = []

        with self.db.connect() as conn:
            with conn.transaction():
                conn.exec(f'''
                    UPDATE 
                        {self.document_table.name}
                    SET
                        deleted=1
                    WHERE
                        filename {_IN(names)}
                ''', names)

                for f in p.files:
                    rec = {
                        'pn': p.personUid,
                        'title': f.title,
                        'mimetype': 'application/pdf',  ## @TODO
                        'username': user_uid,
                        'data': f.data,
                        'filename': f.filename,
                        'size': len(f.data),
                    }

                    uid = conn.insert_one(
                        self.document_table.name,
                        self.document_table.key_column,
                        rec,
                        with_id=True
                    )

                    gws.log.info(f'INSERT pn={p.personUid!r} file={f.filename!r} uid={uid!r}')
                    uids.append(uid)

        return uids

    def _all_documents(self):

        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT
                    {self.document_table.key_column} as uid,
                    pn,
                    title,
                    size,
                    filename,
                    created
                FROM
                    {self.document_table.name}
                WHERE
                    deleted=0
                ORDER BY
                    title
            ''')
            return [
                DocumentProps(
                    uid=str(r['uid']),
                    personUid=str(r['pn']),
                    title=r['title'],
                    size=r['size'],
                    filename=r['filename'],
                    created=r['created'],
                ) for r in rs]


def _IN(ls):
    return 'IN(' + ','.join(['%s'] * len(ls)) + ')'
