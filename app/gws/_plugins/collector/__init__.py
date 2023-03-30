"""Feature collections editor."""

import gws.ext.db.provider.postgres

import gws
import gws.types as t
import gws.base.api
import gws.base.db
import gws.base.model
import gws.base.style
import gws.base.template
import gws.lib.feature
import gws.lib.mime
import gws.lib.style
import gws.base.web.error


##


class ItemProps(gws.lib.feature.Props):
    type: str
    collectionUid: str


class DocumentProps(gws.lib.feature.Props):
    collectionUid: str


class CollectionProps(gws.lib.feature.Props):
    type: str
    items: list[ItemProps]
    documents: list[DocumentProps]


##

class ItemPrototypeConfig(gws.Config):
    type: str
    name: str
    dataModel: gws.base.model.Config
    style: t.Optional[gws.base.style.Config]  #: style for features
    icon: str = ''


class ItemPrototypeProps(gws.Props):
    type: str
    name: str
    dataModel: gws.base.model.ModelProps
    style: gws.base.style.StyleProps
    icon: str


class ItemPrototype(gws.Node):
    def configure(self):
        

        self.db: gws.ext.db.provider.postgres.Object = t.cast(gws.ext.db.provider.postgres.Object, None)
        self.table: gws.SqlTable = t.cast(gws.SqlTable, None)

        self.link_col: str = ''
        self.type_col: str = 'type'

        self.data_model: gws.IDataModel = t.cast(gws.IDataModel, self.create_child('gws.base.model', self.var('dataModel')))

        self.type = self.var('type')
        self.name = self.var('name')
        self.icon = gws.lib.style.parse_icon(self.var('icon'))

        p = self.var('style')
        self.style: gws.IStyle = (
            gws.base.style.from_config(p) if p
            else gws.base.style.from_props(gws.StyleProps(type='css', values=_DEFAULT_STYLE_VALUES)))

    @property
    def props(self):
        return ItemPrototypeProps(
            type=self.type,
            name=self.name,
            dataModel=self.data_model.props,
            style=self.style.props,
            icon=self.icon
        )

    def validate(self, fprops: gws.lib.feature.Props) -> list[gws.AttributeValidationFailure]:
        f = gws.lib.feature.from_props(fprops)
        return self.data_model.validate(f.attributes)

    def save(self, collection_uid: str, fprops: gws.lib.feature.Props):
        f = gws.lib.feature.from_props(fprops)
        f.attributes.append(gws.Attribute(
            name=self.link_col,
            value=collection_uid,
        ))
        f.attributes.append(gws.Attribute(
            name=self.type_col,
            value=self.type,
        ))
        if str(f.uid).isdigit():
            fs = self.db.edit_operation('update', self.table, [f])
        else:
            f.uid = None
            fs = self.db.edit_operation('insert', self.table, [f])
        return fs[0]

    def delete(self, item_uid):
        with self.db.connect() as conn:
            with conn.transaction():
                conn.execute(f'''DELETE FROM 
                    {conn.quote_table(self.table.name)}
                WHERE 
                    {conn.quote_ident(self.table.key_column)} = %s
                ''', item_uid)

    ##


class CollectionPrototypeConfig(gws.Config):
    type: str
    name: str
    db: t.Optional[str]  #: database provider uid
    collectionTable: gws.base.db.SqlTableConfig  #: sql table configuration
    itemTable: gws.base.db.SqlTableConfig  #: sql table configuration
    documentTable: gws.base.db.SqlTableConfig  #: sql table configuration
    dataModel: t.Optional[gws.base.model.Config]
    items: list[ItemPrototypeConfig]
    linkColumn: str = 'collection_id'
    style: t.Optional[gws.base.style.Config]  #: style for collection center point


class CollectionPrototypeProps(gws.Props):
    type: str
    name: str
    dataModel: gws.base.model.ModelProps
    itemPrototypes: list[ItemPrototypeProps]
    style: gws.base.style.StyleProps


class UploadFile(gws.Data):
    data: bytes
    mimeType: str
    title: str
    filename: str


class CollectionPrototype(gws.Node):
    def configure(self):
        

        self.db = t.cast(gws.ext.db.provider.postgres.Object, gws.base.db.require_provider(self, 'gws.ext.db.provider.postgres'))
        self.table = self.db.configure_table(self.var('collectionTable'))
        self.item_table = self.db.configure_table(self.var('itemTable'))
        self.document_table = self.db.configure_table(self.var('documentTable'))

        self.link_col = self.var('linkColumn')
        self.type_col = 'type'

        p = self.var('dataModel') or self.db.table_data_model_config(self.table)
        self.data_model: gws.IDataModel = t.cast(gws.IDataModel, self.create_child('gws.base.model', p))

        self.type = self.var('type')
        self.name = self.var('name')

        self.item_prototypes = []
        for p in self.var('items'):
            ip = t.cast(ItemPrototype, self.create_child(ItemPrototype, p))
            ip.db = self.db
            ip.table = self.item_table
            ip.link_col = self.link_col
            self.item_prototypes.append(ip)

        p = self.var('style')
        self.style: gws.IStyle = (
            gws.base.style.from_config(p) if p
            else gws.base.style.from_props(gws.StyleProps(type='css', values=_DEFAULT_STYLE_VALUES)))

    @property
    def props(self):
        return CollectionPrototypeProps(
            type=self.type,
            name=self.name,
            dataModel=self.data_model.props,
            itemPrototypes=[f.props for f in self.item_prototypes],
            style=self.style.props,
        )

    def validate(self, fprops: gws.lib.feature.Props) -> list[gws.AttributeValidationFailure]:
        f = gws.lib.feature.from_props(fprops)
        return self.data_model.validate(f.attributes)

    def save(self, fprops: gws.lib.feature.Props):
        f = gws.lib.feature.from_props(fprops)
        f.attributes.append(gws.Attribute(
            name=self.type_col,
            value=self.type,
        ))
        if str(f.uid).isdigit():
            fs = self.db.edit_operation('update', self.table, [f])
        else:
            f.uid = None
            fs = self.db.edit_operation('insert', self.table, [f])
        return fs[0]

    def delete(self, collection_uid):
        with self.db.connect() as conn:
            with conn.transaction():
                conn.execute(f'''DELETE FROM 
                    {conn.quote_table(self.item_table.name)}
                WHERE 
                    {conn.quote_ident(self.link_col)} = %s
                ''', [collection_uid])

                conn.execute(f'''DELETE FROM 
                    {conn.quote_table(self.document_table.name)}
                WHERE 
                    {conn.quote_ident(self.link_col)} = %s
                ''', [collection_uid])

                conn.execute(f'''DELETE FROM 
                    {conn.quote_table(self.table.name)}
                WHERE 
                    {conn.quote_ident(self.table.key_column)} = %s
                ''', [collection_uid])

    def delete_item(self, collection_uid, item_uid):
        with self.db.connect() as conn:
            with conn.transaction():
                conn.execute(f'''DELETE FROM 
                    {conn.quote_table(self.item_table.name)}
                WHERE 
                    {conn.quote_ident(self.link_col)} = %s
                    AND {conn.quote_ident(self.item_table.key_column)} = %s
                ''', [collection_uid, item_uid])

    def delete_document(self, collection_uid, document_uid):
        with self.db.connect() as conn:
            with conn.transaction():
                conn.execute(f'''DELETE FROM 
                    {conn.quote_table(self.document_table.name)}
                WHERE 
                    {conn.quote_ident(self.link_col)} = %s
                    AND {conn.quote_ident(self.document_table.key_column)} = %s
                ''', [collection_uid, document_uid])

    def get_collection_ids(self):
        colls = self.db.select(gws.SqlSelectArgs(table=self.table, extra_where=['type=%s', self.type]))
        return [str(c.uid) for c in colls]

    def get_collections(self):
        res = []

        collections = self.db.select(gws.SqlSelectArgs(table=self.table, extra_where=['type=%s', self.type]))
        items = []
        documents = []
        uids = [c.uid for c in collections]
        if uids:
            cond = self.link_col + ' IN (%s)' % ','.join('%s' for _ in collections)

            items = self.db.select(gws.SqlSelectArgs(
                table=self.item_table,
                extra_where=[cond] + uids,
            ))

            documents = self.db.select(gws.SqlSelectArgs(
                table=self.document_table,
                extra_where=[cond] + uids,
                columns=['id', self.link_col, 'title', 'mimetype', 'filename', 'size'],
            ))

        for coll in collections:
            coll.apply_data_model(self.data_model)
            props = coll.props
            props.type = self.type

            props.items = []
            props.documents = []

            for item in items:
                if str(item.attr(self.link_col)) == str(coll.uid):
                    itype = item.attr(self.type_col)
                    ip = self.item_prototype(itype)
                    if ip:
                        item.apply_data_model(ip.data_model)
                        # @TODO
                        for a in item.attributes:
                            if a.type == 'bytes':
                                a.value = None
                        iprops = item.props
                        iprops.type = itype
                        props.items.append(iprops)

            for doc in documents:
                if str(doc.attr(self.link_col)) == str(coll.uid):
                    props.documents.append(doc.props)

            res.append(props)

        return res

    def item_prototype(self, type):
        for ip in self.item_prototypes:
            if ip.type == type:
                return ip

    def upload_documents(self, collection_uid, files: list[UploadFile]):
        with self.db.connect() as conn:
            with conn.transaction():
                for f in files:
                    rec = {
                        self.link_col: collection_uid,
                        'title': f.title or f.filename,
                        'mimetype': gws.lib.mime.for_path(f.filename),
                        'data': f.data,
                        'filename': f.filename,
                        'size': len(f.data),
                    }

                    conn.insert_one(
                        self.document_table.name,
                        self.document_table.key_column,
                        rec,
                    )

    def get_document(self, collection_uid, document_uid):
        documents = self.db.select(gws.SqlSelectArgs(
            table=self.document_table,
            uids=[document_uid]
        ))
        for doc in documents:
            return doc


##


class Config(gws.WithAccess):
    """Collection editor action"""

    collections: list[CollectionPrototypeConfig]


##


class GetPrototypesResponse(gws.Response):
    collectionPrototypes: list[CollectionPrototypeProps]


class GetCollectionsParams(gws.Params):
    type: str


class GetCollectionsResponse(gws.Response):
    collections: list[CollectionProps]


class SaveCollectionParams(gws.Params):
    type: str
    feature: gws.lib.feature.Props


class SaveCollectionResponse(gws.Response):
    collectionUid: str


class ValidationResponse(gws.Response):
    failures: list[gws.AttributeValidationFailure]


class SaveItemParams(gws.Params):
    type: str
    collectionUid: str
    feature: gws.lib.feature.Props


class SaveItemResponse(gws.Response):
    collectionUid: str
    itemUid: str


class DeleteCollectionParams(gws.Params):
    collectionUid: str


class DeleteItemParams(gws.Params):
    collectionUid: str
    itemUid: str


class DeleteDocumentParams(gws.Params):
    collectionUid: str
    documentUid: str


class GetDocumentParams(gws.Params):
    collectionUid: str
    documentUid: str


class UploadDocumentsParams(gws.Params):
    collectionUid: str
    files: list[UploadFile]


class UploadDocumentsResponse(gws.Response):
    pass


##

_DEFAULT_STYLE_VALUES = {
    'fill': 'rgba(0,0,0,1)',
    'stroke': 'rgba(0,0,0,1)',
    'stoke_width': 1,
}


class Object(gws.base.api.Action):

    @property
    def props(self):
        return gws.Props(
            enabled=True,
        )

    def configure(self):
        

        self.db = t.cast(gws.ext.db.provider.postgres.Object, gws.base.db.require_provider(self, 'gws.ext.db.provider.postgres'))

        self.collection_prototypes: list[CollectionPrototype] = []
        for p in self.var('collections'):
            self.collection_prototypes.append(t.cast(CollectionPrototype, self.create_child(CollectionPrototype, p)))

    def api_get_prototypes(self, req: gws.IWebRequest, p: gws.Params) -> GetPrototypesResponse:
        return GetPrototypesResponse(collectionPrototypes=[cp.props for cp in self.collection_prototypes])

    def api_get_collections(self, req: gws.IWebRequest, p: GetCollectionsParams) -> GetCollectionsResponse:
        for cp in self.collection_prototypes:
            if cp.type == p.type:
                return GetCollectionsResponse(collections=cp.get_collections())
        raise gws.base.web.error.NotFound()

    def api_validate_collection(self, req: gws.IWebRequest, p: SaveCollectionParams) -> ValidationResponse:
        for cp in self.collection_prototypes:
            if cp.type == p.type:
                failures = cp.validate(p.feature)
                return ValidationResponse(failures=failures)
        raise gws.base.web.error.NotFound()

    def api_save_collection(self, req: gws.IWebRequest, p: SaveCollectionParams) -> SaveCollectionResponse:
        for cp in self.collection_prototypes:
            if cp.type == p.type:
                coll = cp.save(p.feature)
                return SaveCollectionResponse(collectionUid=coll.uid)
        raise gws.base.web.error.NotFound()

    def api_validate_item(self, req: gws.IWebRequest, p: SaveItemParams) -> ValidationResponse:
        cp, ip = self.collection_and_item(p)
        failures = ip.validate(p.feature)
        return ValidationResponse(failures=failures)

    def api_save_item(self, req: gws.IWebRequest, p: SaveItemParams) -> SaveItemResponse:
        cp, ip = self.collection_and_item(p)
        item = ip.save(p.collectionUid, p.feature)
        return SaveItemResponse(
            collectionUid=p.collectionUid,
            itemUid=item.uid
        )

    def api_delete_collection(self, req: gws.IWebRequest, p: DeleteCollectionParams) -> gws.Response:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.base.web.error.NotFound()
        cp.delete(p.collectionUid)
        return gws.Response()

    def api_delete_item(self, req: gws.IWebRequest, p: DeleteItemParams) -> gws.Response:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.base.web.error.NotFound()

        cp.delete_item(p.collectionUid, p.itemUid)
        return gws.Response()

    def api_upload_documents(self, req: gws.IWebRequest, p: UploadDocumentsParams) -> gws.Response:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.base.web.error.NotFound()
        cp.upload_documents(p.collectionUid, p.files)
        return gws.Response();

    def api_delete_document(self, req: gws.IWebRequest, p: DeleteDocumentParams) -> gws.Response:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.base.web.error.NotFound()

        cp.delete_document(p.collectionUid, p.documentUid)
        return gws.Response()

    def http_get_document(self, req: gws.IWebRequest, p: GetDocumentParams) -> gws.ContentResponse:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.base.web.error.NotFound()

        doc = cp.get_document(p.collectionUid, p.documentUid)
        if not doc:
            raise gws.base.web.error.NotFound()

        return gws.ContentResponse(
            mime=doc.attr('mimetype'),
            content=doc.attr('data'),
            attachment_name=doc.attr('filename')
        )

    def collection_and_item(self, p: SaveItemParams):
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.base.web.error.NotFound()

        ip = cp.item_prototype(p.type)
        if not ip:
            raise gws.base.web.error.NotFound()

        return cp, ip

    def collection_proto_from_collection_uid(self, collection_uid) -> CollectionPrototype:
        for cp in self.collection_prototypes:
            if collection_uid in cp.get_collection_ids():
                return cp
