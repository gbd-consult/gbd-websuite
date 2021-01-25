"""FS-Info."""

import gws
import gws.common.action
import gws.common.db
import gws.common.template
import gws.ext.db.provider.postgres
import gws.ext.helper.alkis
import gws.gis.proj
import gws.gis.shape
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """FSInfo action"""

    db: t.Optional[str]  #: database provider uid
    dataTable: gws.common.db.SqlTableConfig
    documentTable: gws.common.db.SqlTableConfig
    templates: t.Optional[t.List[t.ext.template.Config]]  #: client templates


class FindFlurstueckParams(t.Params):
    gemarkung: str = ''
    flur: str = ''
    flurstueck: str = ''
    nachname: str = ''
    vorname: str = ''
    pn: str = ''


class FindFlurstueckResponse(t.Response):
    features: t.List[t.FeatureProps]
    total: int


class GetGemarkungenResponse(t.Response):
    names: t.List[str]


class RelationProps:
    link: dict
    person: dict
    documents: t.List[dict]


class GetDetailsParams(t.Params):
    fsUid: str


class GetDetailsResponse(t.Response):
    feature: t.FeatureProps
    html: str


class CreateDocumentParams(t.Params):
    data: bytes
    mimeType: str
    personUid: str
    title: str


class UpdateDocumentParams(t.Params):
    data: bytes
    mimeType: str
    documentUid: str


class GetDocumentParams(t.Params):
    documentUid: str


class DocumentResponse(t.Params):
    documentUid: str


class DeleteDocumentParams(t.Params):
    documentUid: str


class DeleteDocumentResponse(t.Params):
    documentUid: str


class DocumentProps(t.Props):
    uid: str
    title: str
    pn: str
    vorname: str
    nachname: str


class GetDocumentsResponse(t.Params):
    documents: t.List[DocumentProps]


class Object(gws.common.action.Object):
    @property
    def props(self):
        return t.Props(enabled=True)

    def configure(self):
        super().configure()

        self.db = t.cast(gws.ext.db.provider.postgres.Object, gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres'))
        self.data_table = self.db.configure_table(self.var('dataTable'))
        self.document_table = self.db.configure_table(self.var('documentTable'))
        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'))
        self.details_template: t.ITemplate = gws.common.template.find(self.templates, subject='fsinfo.details')

    def api_find_flurstueck(self, req: t.IRequest, p: FindFlurstueckParams) -> FindFlurstueckResponse:
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

        fs = self.db.select(t.SelectArgs(
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

    def api_get_gemarkungen(self, req: t.IRequest, p: t.Params) -> GetGemarkungenResponse:
        """Return a list of Gemarkung names"""

        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT DISTINCT gemarkung
                FROM {self.data_table.name}
                ORDER BY gemarkung
            ''')

            return GetGemarkungenResponse(names=[r['gemarkung'] for r in rs])

    def api_get_details(self, req: t.IRequest, p: GetDetailsParams) -> GetDetailsResponse:
        fs_key = self.data_table.key_column

        fs = self.db.select(t.SelectArgs(
            table=self.data_table,
            extra_where=[f'{fs_key} = %s', p.fsUid]
        ))
        if not fs:
            raise gws.web.error.NotFound()

        feature = None
        records = []

        for f in fs:
            if not feature:
                feature = f
            rec = f.attr_dict
            rec['documents'] = []
            records.append(rec)

        for doc in self._get_documents():
            for rec in records:
                if rec['pn'] == doc['pn']:
                    rec['documents'].append(doc)

        html = self.details_template.render({
            'feature': feature,
            'records': records,
        })

        fprops = f.apply_templates(self.templates).props
        del fprops.attributes

        return GetDetailsResponse(feature=fprops, html=html.content)

    def api_create_document(self, req: t.IRequest, p: CreateDocumentParams) -> DocumentResponse:
        uid = self._insert_document(p.personUid, p.title, p, req)
        return DocumentResponse(documentUid=str(uid))

    def api_update_document(self, req: t.IRequest, p: UpdateDocumentParams) -> DocumentResponse:
        doc = self._get_document(p.documentUid)
        if not doc:
            raise gws.web.error.NotFound()

        uid = self._insert_document(doc['pn'], doc['title'], p, req)
        return DocumentResponse(documentUid=str(uid))

    def api_delete_document(self, req: t.IRequest, p: DeleteDocumentParams) -> DeleteDocumentResponse:
        uid = p.documentUid

        with self.db.connect() as conn:
            conn.exec(f'''
                UPDATE {self.document_table.name}
                SET deleted=1
                WHERE {self.document_table.key_column} = %s
            ''', [uid])

            return DeleteDocumentResponse(documentUid=str(uid))

    def api_get_documents(self, req: t.IRequest, p: t.Params) -> GetDocumentsResponse:
        docs = self._get_documents()
        pns = set(str(d['pn']) for d in docs)

        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT DISTINCT pn, nachname, vorname
                FROM {self.data_table.name}
                WHERE pn IN ({','.join(pns)})
            ''')
            pmap = {r['pn']: r for r in rs}

        for doc in docs:
            p = pmap[doc['pn']]
            doc['uid'] = str(doc['uid'])
            doc['nachname'] = p['nachname']
            doc['vorname'] = p['vorname']

        return GetDocumentsResponse(documents=docs)

    def http_get_document(self, req: t.IRequest, p: GetDocumentParams) -> t.HttpResponse:
        doc = self._get_document(p.documentUid)
        if not doc:
            raise gws.web.error.NotFound()
        return t.HttpResponse(
            mime=doc['mimetype'],
            content=doc['data']
        )

    ##

    def _get_document(self, document_uid):
        with self.db.connect() as conn:
            return conn.select_one(f'''
                SELECT * FROM {self.document_table.name} 
                    WHERE {self.document_table.key_column} = %s
            ''', [document_uid])

    def _get_documents(self):
        with self.db.connect() as conn:
            rs = conn.select(f'''
                SELECT
                    *,
                    {self.document_table.key_column} as uid,
                    '' AS data
                FROM
                    {self.document_table.name}
                WHERE
                    deleted=0
                ORDER BY
                    title,
                    created DESC
            ''')

            # ignore previous versions of the same doc (pers-id+title)
            seen = set()
            docs = []

            for r in rs:
                key = (r['pn'], r['title'])
                if key not in seen:
                    docs.append(r)
                    seen.add(key)

            return docs

    def _insert_document(self, person_uid, title, p, req):
        rec = {
            'pn': person_uid,
            'title': title,
            'mimetype': 'application/pdf',  ## @TODO
            'username': req.user.uid,
            'data': p.data,
        }

        with self.db.connect() as conn:
            uid = conn.insert_one(
                self.document_table.name,
                self.document_table.key_column,
                rec,
                with_id=True
            )
            return uid
