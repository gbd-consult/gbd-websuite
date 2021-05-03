import re
import os

import gws
import gws.web
import gws.tools.date
import gws.common.model
import gws.gis.shape
import gws.tools.net
import gws.tools.misc
import gws.ext.db.provider.postgres
import gws.common.action
import gws.common.db
import gws.gis.feature
import gws.web.error

import gws.types as t


class TableConfig(t.Config):
    title: t.Optional[str]
    table: gws.common.db.SqlTableConfig  #: sql table configurations
    sort: t.Optional[str]  #: sort expression
    dataModel: t.Optional[gws.common.model.Config]  #: table data model
    widths: t.Optional[t.List[int]]  #: column widths, 0 to exclude
    withFilter: t.Optional[bool]  #: use filter boxes
    disableAddButton: t.Optional[bool]  #: disable the 'add' button


class Config(t.WithTypeAndAccess):
    """Table Editor action"""

    db: t.Optional[str]  #: database provider uid
    tables: t.List[TableConfig]


class TableProps(t.Props):
    uid: str
    title: str


class GetTablesResponse(t.Response):
    tables: t.List[TableProps]


class LoadDataParams(t.Params):
    tableUid: str


class LoadDataResponse(t.Response):
    tableUid: str
    key: str
    attributes: t.List[t.Attribute]
    records: t.List[t.Any]
    widths: t.Optional[t.List[int]]
    withFilter: bool
    withAdd: bool
    withDelete: bool


class SaveDataParams(t.Params):
    tableUid: str
    attributes: t.List[t.Attribute]
    records: t.List[t.Any]


class SaveDataResponse(t.Response):
    pass


class Table(t.Data):
    uid: str
    title: str
    table: t.SqlTable
    data_model: t.IModel
    widths: t.List[int]
    with_filter: bool


class Object(gws.common.action.Object):
    @property
    def props(self):
        return t.Props(enabled=True)

    def configure(self):
        super().configure()

        self.db: gws.ext.db.provider.postgres.Object = t.cast(
            gws.ext.db.provider.postgres.Object,
            gws.common.db.require_provider(self, 'gws.ext.db.provider.postgres'))

        self.tables: t.List[Table] = []

        for p in self.var('tables'):
            table = self.db.configure_table(p.table)
            m = p.dataModel or self.db.table_data_model_config(table)
            self.tables.append(Table(
                uid=p.uid or gws.as_uid(table.name),
                title=p.title or table.name,
                table=table,
                data_model=t.cast(t.IModel, self.create_child('gws.common.model', m)),
                sort=p.sort or table.key_column,
                widths=p.widths or [],
                with_filter=bool(p.withFilter),
                with_add=not bool(p.disableAddButton),
                with_delete=False,
            ))

    def api_get_tables(self, req: t.IRequest, p: t.Params) -> GetTablesResponse:
        return GetTablesResponse(
            tables=[t.Props(
                uid=tbl.uid,
                title=tbl.title,
            ) for tbl in self.tables]
        )

    def api_load_data(self, req: t.IRequest, p: LoadDataParams) -> LoadDataResponse:
        tbl = self._get_table(p.tableUid)
        if not tbl:
            raise gws.web.error.NotFound()

        features = self.db.select(t.SelectArgs(
            table=tbl.table,
            sort=tbl.sort,
            extra_where=['true']
        ))

        attributes = None
        records = []

        for f in features:
            f.apply_data_model(tbl.data_model)
            records.append([a.value for a in f.attributes])
            if not attributes:
                for a in f.attributes:
                    del a.value
                attributes = f.attributes

        return LoadDataResponse(
            tableUid=tbl.uid,
            key=tbl.table.key_column,
            attributes=attributes,
            records=records,
            widths=tbl.widths or None,
            withFilter=tbl.with_filter,
            withAdd=tbl.with_add,
            withDelete=tbl.with_delete,
        )

    def api_save_data(self, req: t.IRequest, p: SaveDataParams) -> SaveDataResponse:
        tbl = self._get_table(p.tableUid)
        if not tbl:
            raise gws.web.error.NotFound()

        upd_features = []
        ins_features = []

        for rec in p.records:
            atts = []
            uid = None

            for a, v in zip(p.attributes, rec):
                # Normalize empty inputs to insert NULL into the database.
                if v == '':
                    v = None
                atts.append(t.Attribute(name=a.name, value=v))
                if a.name == tbl.table.key_column:
                    uid = v

            f = gws.gis.feature.from_props(t.FeatureProps(uid=uid, attributes=atts))
            if uid:
                upd_features.append(f)
            else:
                ins_features.append(f)

        if ins_features and not tbl.with_add:
            # @TODO: this must be done in the dataModel
            raise gws.web.error.Forbidden()

        if upd_features:
            self.db.edit_operation('update', tbl.table, upd_features)
        if ins_features:
            self.db.edit_operation('insert', tbl.table, ins_features)

        return SaveDataResponse()

    def _get_table(self, table_uid: str):
        for tbl in self.tables:
            if tbl.uid == table_uid:
                return tbl
