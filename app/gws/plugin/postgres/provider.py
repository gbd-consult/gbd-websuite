import gws
import gws.base.db
import gws.base.model
import gws.gis.crs
import gws.base.feature
import gws.base.shape
import gws.lib.sql.postgres
import gws.types as t


@gws.ext.config.db('postgres')
class Config(gws.Config):
    """Postgres/Postgis database provider"""

    database: str = ''  #: database name
    host: str = 'localhost'  #: database host
    password: str  #: password
    port: int = 5432  #: database port
    timeout: gws.Duration = '0'  #: query timeout
    connectTimeout: gws.Duration = '0'  #: connect timeout
    user: str  #: username


@gws.ext.object.db('postgres')
class Object(gws.Node, gws.IDatabase):
    def configure(self):

        def ping():
            gws.log.debug(f'db: ping {self.uid!r}')
            try:
                with self.connection() as conn:
                    conn.select_value('select 1 + 1')
                    gws.log.debug(f'db connection {self.uid!r}: ok')
            except gws.lib.sql.postgres.Error as exc:
                raise gws.Error(f'cannot open db connection {self.uid!r}') from exc

        gws.get_app_global(f'db_ping_{self.uid}', ping)

    def connection(self) -> gws.lib.sql.postgres.Connection:
        return gws.lib.sql.postgres.Connection(vars(self.config))

    def describe(self, table: gws.SqlTable) -> t.List[gws.SqlColumn]:

        def _get():
            with self.connection() as conn:
                return conn.columns(table.name)

        key = self.class_name + '_describe_' + table.name
        return gws.get_server_global(key, _get)

    def configure_table(self, cfg: gws.base.db.SqlTableConfig) -> gws.SqlTable:
        table = gws.SqlTable(name=cfg.get('name'))
        cols = self.describe(table)

        if not cols:
            raise gws.Error(f'table {table.name!r} not found or not accessible')

        cname = cfg.get('keyColumn')
        if cname:
            for col in cols:
                if col.name == cname:
                    table.key_column = col
                    break
            if not table.key_column:
                raise gws.Error(f'invalid keyColumn {cname!r} for table {table.name!r}')
        else:
            cs = [col for col in cols if col.is_key]
            if len(cs) == 1:
                table.key_column = cs[0]

        cname = cfg.get('geometryColumn')
        if cname:
            for col in cols:
                if col.name == cname and col.is_geometry:
                    table.geometry_column = col
                    break
            if not table.geometry_column:
                raise gws.Error(f'invalid geometryColumn {cname!r} for table {table.name!r}')
        else:
            cs = [col for col in cols if col.is_geometry]
            if len(cs) == 1:
                table.geometry_column = cs[0]
            if len(cs) > 1:
                table.geometry_column = cs[0]
                gws.log.debug(f'found multiple geometry columns for table {table.name!r}, using {table.geometry_column.name!r}')

        cname = cfg.get('searchColumn')
        if cname:
            for col in cols:
                if col.name == cname:
                    table.search_column = col
                    break
            if not table.search_column:
                raise gws.Error(f'invalid searchColumn {cname!r} for table {table.name!r}')

        return table

    def select_features(self, args: gws.SqlSelectArgs) -> t.List[gws.IFeature]:

        where = []

        if args.keyword and args.table.search_column:
            # @TODO search mode (startsWith, contains, exact etc)
            where.append(gws.Sql('{:name} ILIKE {:like}', args.table.search_column.name, ['*%', args.keyword]))

        if args.shape and args.table.geometry_column:
            # @TODO search mode (intesects, contains etc)
            crs = gws.gis.crs.get(args.table.geometry_column.srid)
            shape = args.shape.tolerance_polygon(args.geometry_tolerance).transformed_to(crs)
            where.append(gws.Sql('ST_Intersects({:value}::geometry, {:name})', shape.ewkb_hex, args.table.geometry_column.name))

        if args.uids:
            if not args.table.key_column:
                return []
            where.append(gws.Sql('{:name} IN ({:values})', args.table.key_column.name, args.uids))

        if args.extra_where:
            where.append(args.extra_where)

        if not where:
            return []

        sql_text = ['SELECT']
        sql_args = []

        if args.columns:
            sql_text.append('{:names}')
            sql_args.append(args.columns)
        else:
            sql_text.append('*')

        sql_text.append('FROM {:qname}')
        sql_args.append(args.table.name)

        sql_text.append('WHERE {:and}')
        sql_args.append(where)

        p = args.sort
        if p:
            sql_text.append('ORDER BY {:name}')
            sql_args.append(p)

        p = args.limit
        if p:
            sql_text.append('LIMIT {:int}')
            sql_args.append(p)

        with self.connection() as conn:
            gws.log.debug(f'SELECT_FEATURES_START {sql_text} p={sql_args}')
            recs = conn.select(' '.join(sql_text), *sql_args)
            gws.log.debug(f'SELECT_FEATURES_END len={len(recs)}')

        return [self.feature_from_record(args.table, r) for r in recs]

    def insert_features(self, table: gws.SqlTable, features: t.List[gws.IFeature], on_conflict: gws.Sql = None) -> t.List[gws.IFeature]:
        return self._insert_or_update(table, features, on_conflict, is_insert=True)

    def update_features(self, table: gws.SqlTable, features: t.List[gws.IFeature]) -> t.List[gws.IFeature]:
        return self._insert_or_update(table, features, None, is_insert=False)

    def _insert_or_update(self, table, features, on_conflict, is_insert):
        uids = []
        recs = [self.record_from_feature(table, f) for f in features]

        with self.connection() as conn:
            with conn.transaction():
                for rec in recs:
                    if is_insert:
                        uids.append(conn.insert(table.name, rec, table.key_column.name, on_conflict=on_conflict))
                    else:
                        conn.update(table.name, rec, table.key_column.name)
                        uids.append(rec.get(table.key_column.name))
            if not uids:
                return []
            recs = conn.select('SELECT * FROM {:qname} WHERE {:name} IN {{:values})', table.name, table.key_column.name, uids)
            return [self.feature_from_record(table, rec) for rec in recs]

    def delete_features(self, table: gws.SqlTable, features: t.List[gws.IFeature]):
        uids = [f.uid for f in features]
        with self.connection() as conn:
            with conn.transaction():
                conn.delete_many(table.name, table.key_column.name, uids)

    def table_data_model_config(self, table: gws.SqlTable) -> gws.Config:
        rules = []

        for col in self.describe(table):
            if col.is_geometry:
                continue
            rules.append(gws.base.model.Rule(
                title=col.name,
                name=col.name,
                source=col.name,
                type=col.type,
                editable=not col.is_key,
            ))

        return gws.Config(
            rules=rules,
            geometryType=table.geometry_column.type if table.geometry_column else None,
            crs=table.geometry_column.srid if table.geometry_column else None,
        )

    def record_from_feature(self, table: gws.SqlTable, feature: gws.IFeature) -> dict:
        rec = {a.name: a.value for a in feature.attributes}

        if table.key_column:
            rec[table.key_column] = feature.uid

        if table.geometry_column and feature.shape:
            crs = gws.gis.crs.get(table.geometry_column.srid)
            shape = feature.shape.to_type(table.geometry_column.gtype).transformed_to(crs)
            rec[table.geometry_column] = shape.ewkb_hex

        return rec

    def feature_from_record(self, table: gws.SqlTable, rec: dict) -> gws.IFeature:
        shape = None
        if table.geometry_column:
            g = rec.pop(table.geometry_column.name, None)
            if g:
                # assuming geometries are returned in hex
                crs = gws.gis.crs.get(table.geometry_column.srid)
                shape = gws.base.shape.from_wkb_hex(g, crs)

        uid = None
        if table.key_column:
            uid = str(rec.get(table.key_column.name, None))
        if not uid:
            uid = gws.sha256(rec)

        return gws.base.feature.Feature(uid=uid, attributes=rec, shape=shape)


##


def create(root: gws.IRoot, cfg, parent: gws.Node = None, shared: bool = False) -> Object:
    key = gws.pick(cfg, 'host', 'port', 'user', 'database')
    return root.create(Object, cfg, parent, shared, key)


def require_for(obj: gws.INode) -> Object:
    uid = obj.var('db')
    if uid:
        prov = obj.root.find(klass=Object, uid=uid)
        if not prov:
            raise gws.Error(f'{obj.uid}: db provider {uid!r} not found')
    else:
        prov = obj.root.find(klass=Object)
        if not prov:
            raise gws.Error(f'{obj.uid}: db provider postgres not found')
    return prov
