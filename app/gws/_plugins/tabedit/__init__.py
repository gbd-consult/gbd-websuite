import gws.ext.db.provider.postgres

import gws
import gws.types as t
import gws.base.api
import gws.base.db
import gws.base.model
import gws.lib.feature
import gws.lib.shape
import gws.lib.date
import gws.lib.misc
import gws.lib.net
import gws.base.web
import gws.base.web.error


class TableConfig(gws.Config):
    title: t.Optional[str]
    table: gws.base.db.SqlTableConfig  #: sql table configuration
    sort: t.Optional[str]  #: sort expression
    dataModel: t.Optional[gws.base.model.Config]  #: table data model
    widths: t.Optional[list[int]]  #: column widths, 0 to exclude
    withFilter: t.Optional[bool]  #: use filter boxes
    withAdd: t.Optional[bool]  #: use the 'add' button
    # @TODO
    # withDelete: t.Optional[bool]  #: use the 'delete' button


class Config(gws.WithAccess):
    """Table Editor action"""

    db: t.Optional[str]  #: database provider uid
    tables: list[TableConfig]


class TableProps(gws.Props):
    uid: str
    title: str


class GetTablesResponse(gws.Response):
    tables: list[TableProps]


class LoadDataParams(gws.Params):
    tableUid: str


class LoadDataResponse(gws.Response):
    tableUid: str
    key: str
    attributes: list[gws.Attribute]
    records: list[t.Any]
    widths: t.Optional[list[int]]
    withFilter: bool
    withAdd: bool
    withDelete: bool


class SaveDataParams(gws.Params):
    tableUid: str
    attributes: list[gws.Attribute]
    records: list[t.Any]


class SaveDataResponse(gws.Response):
    pass


class Table(gws.Data):
    uid: str
    title: str
    table: gws.SqlTable
    data_model: gws.IDataModel
    widths: list[int]
    with_filter: bool


class Object(gws.base.api.Action):
    @property
    def props(self):
        return gws.Props(enabled=True)

    def configure(self):
        

        self.db: gws.ext.db.provider.postgres.Object = t.cast(
            gws.ext.db.provider.postgres.Object,
            gws.base.db.require_provider(self, 'gws.ext.db.provider.postgres'))

        self.tables: list[Table] = []

        for p in self.var('tables'):
            table = self.db.configure_table(p.table)
            m = p.dataModel or self.db.table_data_model_config(table)
            self.tables.append(Table(
                uid=p.uid or gws.as_uid(table.name),
                title=p.title or table.name,
                table=table,
                data_model=t.cast(gws.IDataModel, self.create_child('gws.base.model', m)),
                sort=p.sort or table.key_column,
                widths=p.widths or [],
                with_filter=bool(p.withFilter),
                with_add=bool(p.withAdd),
                with_delete=False,
            ))

    def api_get_tables(self, req: gws.IWebRequest, p: gws.Params) -> GetTablesResponse:
        return GetTablesResponse(
            tables=[gws.Props(
                uid=tbl.uid,
                title=tbl.title,
            ) for tbl in self.tables]
        )

    def api_load_data(self, req: gws.IWebRequest, p: LoadDataParams) -> LoadDataResponse:
        tbl = self._get_table(p.tableUid)
        if not tbl:
            raise gws.base.web.error.NotFound()

        features = self.db.select(gws.SqlSelectArgs(
            table=tbl.table,
            sort=tbl.sort,
            extra_where=['true']
        ))

        attributes = [
            gws.Attribute(
                name=r.name,
                title=r.title,
                type=r.type,
                editable=r.editable,
            ) for r in tbl.data_model.rules
        ]

        records = []

        for f in features:
            f.apply_data_model(tbl.data_model)
            attr_dict = f.attr_dict
            records.append([attr_dict.get(a.name) for a in attributes])

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

    def api_save_data(self, req: gws.IWebRequest, p: SaveDataParams) -> SaveDataResponse:
        tbl = self._get_table(p.tableUid)
        if not tbl:
            raise gws.base.web.error.NotFound()

        upd_features = []
        ins_features = []

        for rec in p.records:
            atts = []
            uid = None

            for a, v in zip(p.attributes, rec):
                if v is None or v == '':
                    continue
                atts.append(gws.Attribute(name=a.name, value=v))
                if a.name == tbl.table.key_column:
                    uid = v

            f = gws.lib.feature.from_props(gws.lib.feature.Props(uid=uid, attributes=atts))
            if uid:
                upd_features.append(f)
            else:
                ins_features.append(f)

        if ins_features and not tbl.with_add:
            # @TODO: this must be done in the dataModel
            raise gws.base.web.error.Forbidden()

        if upd_features:
            self.db.edit_operation('update', tbl.table, upd_features)
        if ins_features:
            self.db.edit_operation('insert', tbl.table, ins_features)

        return SaveDataResponse()

    def _get_table(self, table_uid: str):
        for tbl in self.tables:
            if tbl.uid == table_uid:
                return tbl
