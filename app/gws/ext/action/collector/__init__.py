"""Feature collections editor."""

import gws.types as t

import gws.common.action
import gws.common.db
import gws.common.model
import gws.common.style
import gws.common.template
import gws.tools.style
import gws.gis.feature
import gws.web.error
import gws.ext.db.provider.postgres


##


class ItemProps(t.FeatureProps):
    type: str
    collectionUid: str


class CollectionProps(t.FeatureProps):
    type: str
    items: t.List[ItemProps]


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
            self.db.edit_operation('update', self.table, [f])
        else:
            f.uid = None
            self.db.edit_operation('insert', self.table, [f])

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
    dataModel: t.Optional[gws.common.model.Config]
    items: t.List[ItemPrototypeConfig]
    linkColumn: str = 'collection_id'


class CollectionPrototypeProps(t.Props):
    type: str
    name: str
    dataModel: gws.common.model.ModelProps
    itemPrototypes: t.List[ItemPrototypeProps]


class CollectionPrototype(gws.Object):
    def configure(self):
        super().configure()

        self.db = t.cast(gws.ext.db.provider.postgres.Object, gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres'))
        self.table = self.db.configure_table(self.var('collectionTable'))
        self.item_table = self.db.configure_table(self.var('itemTable'))

        self.link_col = self.var('linkColumn')
        self.type_col = 'type'

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

    @property
    def props(self):
        return CollectionPrototypeProps(
            type=self.type,
            name=self.name,
            dataModel=self.data_model.props,
            itemPrototypes=[f.props for f in self.item_prototypes]
        )

    def save(self, fprops: t.FeatureProps):
        f = gws.gis.feature.from_props(fprops)
        f.attributes.append(t.Attribute(
            name=self.type_col,
            value=self.type,
        ))
        if str(f.uid).isdigit():
            self.db.edit_operation('update', self.table, [f])
        else:
            f.uid = None
            self.db.edit_operation('insert', self.table, [f])

    def delete(self, collection_uid):
        with self.db.connect() as conn:
            with conn.transaction():
                conn.execute(f'''DELETE FROM 
                    {conn.quote_table(self.item_table.name)}
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

    def get_object_ids(self):
        colls = self.db.select(t.SelectArgs(table=self.table, extra_where=['type=%s', self.type]))
        return [str(c.uid) for c in colls]

    def get_objects(self):
        res = []

        colls = self.db.select(t.SelectArgs(table=self.table, extra_where=['type=%s', self.type]))
        items = []
        uids = [c.uid for c in colls]
        if uids:
            cond = self.link_col + ' IN (%s)' % ','.join('%s' for _ in colls)
            items = self.db.select(t.SelectArgs(table=self.item_table, extra_where=[cond] + uids))

        for c in colls:
            c.apply_data_model(self.data_model)
            props = c.props
            props.type = self.type

            props.items = []

            for item in items:
                if str(item.attr(self.link_col)) == str(c.uid):
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

            res.append(props)

        return res

    def item_prototype(self, type):
        for ip in self.item_prototypes:
            if ip.type == type:
                return ip


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


class SaveItemParams(t.Params):
    type: str
    collectionUid: str
    feature: t.FeatureProps


class DeleteCollectionParams(t.Params):
    collectionUid: str


class DeleteItemParams(t.Params):
    collectionUid: str
    itemUid: str


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
                return GetCollectionsResponse(collections=cp.get_objects())
        raise gws.web.error.NotFound()

    def api_save_collection(self, req: t.IRequest, p: SaveCollectionParams) -> t.Response:
        for cp in self.collection_prototypes:
            if cp.type == p.type:
                cp.save(p.feature)
                return t.Response()
        raise gws.web.error.NotFound()

    def api_save_item(self, req: t.IRequest, p: SaveItemParams) -> t.Response:
        cp = self.collection_proto_from_collection_uid(p.collectionUid)
        if not cp:
            raise gws.web.error.NotFound()

        ip = cp.item_prototype(p.type)
        if not ip:
            raise gws.web.error.NotFound()

        ip.save(p.collectionUid, p.feature)
        return t.Response()

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

    def collection_proto_from_collection_uid(self, collection_uid) -> CollectionPrototype:
        for cp in self.collection_prototypes:
            if collection_uid in cp.get_object_ids():
                return cp
