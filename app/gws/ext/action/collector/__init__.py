"""Feature collections editor."""

import gws.types as t

import gws.common.action
import gws.common.db
import gws.common.model
import gws.common.style
import gws.common.template
import gws.tools.style
import gws.tools.mime
import gws.gis.feature
import gws.web.error
import gws.ext.db.provider.postgres


##


class ItemProps(t.FeatureProps):
    type: str
    collectionUid: str


class DocumentProps(t.FeatureProps):
    collectionUid: str


class CollectionProps(t.FeatureProps):
    type: str
    items: t.List[ItemProps]
    documents: t.List[DocumentProps]


##

class ItemPrototypeConfig(t.Config):
    type: str
    name: str
    dataModel: gws.common.model.Config
    style: t.Optional[gws.common.style.Config]  #: style for features
    icon: str = ''


class ItemPrototypeProps(t.Props):
    type: str
    name: str
    dataModel: gws.common.model.ModelProps
    style: gws.common.style.StyleProps
    icon: str


class ItemPrototype(gws.Object):
    def configure(self):
        super().configure()

        self.db: gws.ext.db.provider.postgres.Object = t.cast(gws.ext.db.provider.postgres.Object, None)
        self.table: t.SqlTable = t.cast(t.SqlTable, None)

        self.link_col: str = ''
        self.type_col: str = 'type'

        self.data_model: t.IModel = t.cast(t.IModel, self.create_child('gws.common.model', self.var('dataModel')))

        self.type = self.var('type')
        self.name = self.var('name')
        self.icon = gws.tools.style.parse_icon(self.var('icon'))

        p = self.var('style')
        self.style: t.IStyle = (
            gws.common.style.from_config(p) if p
            else gws.common.style.from_props(t.StyleProps(type='css', values=_DEFAULT_STYLE_VALUES)))

    @property
    def props(self):
        return ItemPrototypeProps(
            type=self.type,
            name=self.name,
            dataModel=self.data_model.props,
            style=self.style.props,
            icon=self.icon
        )

    def validate(self, fprops: t.FeatureProps) -> t.List[t.AttributeValidationFailure]:
        f = gws.gis.feature.from_props(fprops)
        return self.data_model.validate(f.attributes)

    def save(self, collection_uid: str, fprops: t.FeatureProps):
        f = gws.gis.feature.from_props(fprops)
        f.attributes.append(t.Attribute(
            name=self.link_col,
            value=collection_uid,
        ))
        f.attributes.append(t.Attribute(
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


class CollectionPrototypeConfig(t.Config):
    type: str
    name: str
    db: t.Optional[str]  #: database provider uid
    collectionTable: gws.common.db.SqlTableConfig  #: sql table configuration
    itemTable: gws.common.db.SqlTableConfig  #: sql table configuration
    documentTable: gws.common.db.SqlTableConfig  #: sql table configuration
    dataModel: t.Optional[gws.common.model.Config]
    items: t.List[ItemPrototypeConfig]
    linkColumn: str = 'collection_id'
    style: t.Optional[gws.common.style.Config]  #: style for collection center point
    hideExpired: bool = False #: hide expired collections


class CollectionPrototypeProps(t.Props):
    type: str
    name: str
    dataModel: gws.common.model.ModelProps
    itemPrototypes: t.List[ItemPrototypeProps]
    style: gws.common.style.StyleProps


class UploadFile(t.Data):
    data: bytes
    mimeType: str
    title: str
    filename: str


class CollectionPrototype(gws.Object):
    def configure(self):
        super().configure()

        self.db = t.cast(gws.ext.db.provider.postgres.Object, gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres'))
        self.table = self.db.configure_table(self.var('collectionTable'))
        self.item_table = self.db.configure_table(self.var('itemTable'))
        self.document_table = self.db.configure_table(self.var('documentTable'))
        self.hide_expired = self.var('hideExpired')

        self.link_col = self.var('linkColumn')
        self.type_col = 'type'
        self.time_start_col = 'beginn'
        self.time_end_col = 'ende'

        p = self.var('dataModel') or self.db.table_data_model_config(self.table)
        self.data_model: t.IModel = t.cast(t.IModel, self.create_child('gws.common.model', p))

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
        self.style: t.IStyle = (
            gws.common.style.from_config(p) if p
            else gws.common.style.from_props(t.StyleProps(type='css', values=_DEFAULT_STYLE_VALUES)))

    @property
    def props(self):
        return CollectionPrototypeProps(
            type=self.type,
            name=self.name,
            dataModel=self.data_model.props,
            itemPrototypes=[f.props for f in self.item_prototypes],
            style=self.style.props,
        )

    def validate(self, fprops: t.FeatureProps) -> t.List[t.AttributeValidationFailure]:
        f = gws.gis.feature.from_props(fprops)
        return self.data_model.validate(f.attributes)

    def save(self, fprops: t.FeatureProps):
        f = gws.gis.feature.from_props(fprops)
        f.attributes.append(t.Attribute(
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

    def get_collection_rows(self):
        extra_where = ['type=%s', self.type]
        if self.hide_expired:
            extra_where[0] += f' AND COALESCE({self.time_end_col}, CURRENT_DATE) >= CURRENT_DATE'
        return self.db.select(t.SelectArgs(table=self.table, extra_where=extra_where))

    def get_collection_ids(self):
        return [str(c.uid) for c in self.get_collection_rows()]

    def get_collections(self):
        res = []

        collections = self.get_collection_rows()
        items = []
        documents = []
        uids = [c.uid for c in collections]
        if uids:
            cond = self.link_col + ' IN (%s)' % ','.join('%s' for _ in collections)

            items = self.db.select(t.SelectArgs(
                table=self.item_table,
                extra_where=[cond] + uids,
            ))

            documents = self.db.select(t.SelectArgs(
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

    def upload_documents(self, collection_uid, files: t.List[UploadFile]):
        with self.db.connect() as conn:
            with conn.transaction():
                for f in files:
                    rec = {
                        self.link_col: collection_uid,
                        'title': f.title or f.filename,
                        'mimetype': gws.tools.mime.for_path(f.filename),
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
        documents = self.db.select(t.SelectArgs(
            table=self.document_table,
            uids=[document_uid]
        ))
        for doc in documents:
            return doc


##


class Config(t.WithTypeAndAccess):
    """Collection editor action"""

    collections: t.List[CollectionPrototypeConfig]


##


class GetPrototypesResponse(t.Response):
    collectionPrototypes: t.List[CollectionPrototypeProps]


class GetCollectionsParams(t.Params):
    type: str


class GetCollectionsResponse(t.Response):
    collections: t.List[CollectionProps]


class SaveCollectionParams(t.Params):
    type: str
    feature: t.FeatureProps


class SaveCollectionResponse(t.Response):
    collectionUid: str


class ValidationResponse(t.Response):
    failures: t.List[t.AttributeValidationFailure]


class SaveItemParams(t.Params):
    type: str
    collectionUid: str
    feature: t.FeatureProps


class SaveItemResponse(t.Response):
    collectionUid: str
    itemUid: str


class DeleteCollectionParams(t.Params):
    collectionUid: str


class DeleteItemParams(t.Params):
    collectionUid: str
    itemUid: str


class DeleteDocumentParams(t.Params):
    collectionUid: str
    documentUid: str


class GetDocumentParams(t.Params):
    collectionUid: str
    documentUid: str


class UploadDocumentsParams(t.Params):
    collectionUid: str
    files: t.List[UploadFile]


class UploadDocumentsResponse(t.Response):
    pass


##

_DEFAULT_STYLE_VALUES = {
    'fill': 'rgba(0,0,0,1)',
    'stroke': 'rgba(0,0,0,1)',
    'stoke_width': 1,
}


class Object(gws.common.action.Object):

    @property
    def props(self):
        return t.Props(
            enabled=True,
        )

    def configure(self):
        super().configure()

        self.db = t.cast(gws.ext.db.provider.postgres.Object, gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres'))

        self.collection_prototypes: t.List[CollectionPrototype] = []
        for p in self.var('collections'):
            self.collection_prototypes.append(t.cast(CollectionPrototype, self.create_child(CollectionPrototype, p)))

    def api_get_prototypes(self, req: t.IRequest, p: t.Params) -> GetPrototypesResponse:
        return GetPrototypesResponse(collectionPrototypes=[cp.props for cp in self.collection_prototypes])

    def api_get_collections(self, req: t.IRequest, p: GetCollectionsParams) -> GetCollectionsResponse:
        for cp in self.collection_prototypes:
            if cp.type == p.type:
                return GetCollectionsResponse(collections=cp.get_collections())
        raise gws.web.error.NotFound()

    def api_validate_collection(self, req: t.IRequest, p: SaveCollectionParams) -> ValidationResponse:
        for cp in self.collection_prototypes:
            if cp.type == p.type:
                failures = cp.validate(p.feature)
                return ValidationResponse(failures=failures)
        raise gws.web.error.NotFound()

    def api_save_collection(self, req: t.IRequest, p: SaveCollectionParams) -> SaveCollectionResponse:
        for cp in self.collection_prototypes:
            if cp.type == p.type:
                coll = cp.save(p.feature)
                return SaveCollectionResponse(collectionUid=coll.uid)
        raise gws.web.error.NotFound()

    def api_validate_item(self, req: t.IRequest, p: SaveItemParams) -> ValidationResponse:
        cp, ip = self.collection_and_item(p)
        failures = ip.validate(p.feature)
        return ValidationResponse(failures=failures)

    def api_save_item(self, req: t.IRequest, p: SaveItemParams) -> SaveItemResponse:
        cp, ip = self.collection_and_item(p)
        item = ip.save(p.collectionUid, p.feature)
        return SaveItemResponse(
            collectionUid=p.collectionUid,
            itemUid=item.uid
        )

    def api_delete_collection(self, req: t.IRequest, p: DeleteCollectionParams) -> t.Response:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.web.error.NotFound()
        cp.delete(p.collectionUid)
        return t.Response()

    def api_delete_item(self, req: t.IRequest, p: DeleteItemParams) -> t.Response:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.web.error.NotFound()

        cp.delete_item(p.collectionUid, p.itemUid)
        return t.Response()

    def api_upload_documents(self, req: t.IRequest, p: UploadDocumentsParams) -> t.Response:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.web.error.NotFound()
        cp.upload_documents(p.collectionUid, p.files)
        return t.Response();

    def api_delete_document(self, req: t.IRequest, p: DeleteDocumentParams) -> t.Response:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.web.error.NotFound()

        cp.delete_document(p.collectionUid, p.documentUid)
        return t.Response()

    def http_get_document(self, req: t.IRequest, p: GetDocumentParams) -> t.FileResponse:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.web.error.NotFound()

        doc = cp.get_document(p.collectionUid, p.documentUid)
        if not doc:
            raise gws.web.error.NotFound()

        return t.FileResponse(
            mime=doc.attr('mimetype'),
            content=doc.attr('data'),
            attachment_name=doc.attr('filename')
        )

    def collection_and_item(self, p: SaveItemParams):
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.web.error.NotFound()

        ip = cp.item_prototype(p.type)
        if not ip:
            raise gws.web.error.NotFound()

        return cp, ip

    def collection_proto_from_collection_uid(self, collection_uid) -> CollectionPrototype:
        for cp in self.collection_prototypes:
            if collection_uid in cp.get_collection_ids():
                return cp
