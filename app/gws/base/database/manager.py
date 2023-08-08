"""Core database utilities."""

import gws
import gws.gis.crs
import gws.lib.sa as sa

import gws.types as t

from . import session as session_module


class Config(gws.Config):
    """Database configuration"""

    providers: list[gws.ext.config.databaseProvider]
    """database providers"""


class Object(gws.Node, gws.IDatabaseManager):
    modelMap: dict[str, gws.IDatabaseModel]
    providerMap: dict[str, gws.IDatabaseProvider]
    rt: '_SaRuntime'

    def __getstate__(self):
        return gws.omit(vars(self), 'rt')

    def configure(self):
        self.rt = _SaRuntime(self)

        self.providerMap = {}
        for cfg in self.cfg('providers', default=[]):
            prov = self.create_provider(cfg)
            self.providerMap[prov.uid] = prov

        self.modelMap = {}
        self.root.app.register_middleware('db', self)

    def post_configure(self):
        self.rt.create_orm()
        self.close_sessions()

    def activate(self):
        self.rt = _SaRuntime(self)
        self.rt.create_orm()

    ##

    def enter_middleware(self, req: gws.IWebRequester):
        pass

    def exit_middleware(self, req: gws.IWebRequester, res: gws.IWebResponder):
        self.close_sessions()

    ##

    def create_provider(self, cfg, **kwargs):
        return self.root.create_shared(gws.ext.object.databaseProvider, cfg, _defaultManager=self, **kwargs)

    def providers(self):
        return list(self.providerMap.values())

    def provider(self, uid):
        return self.providerMap.get(uid)

    def first_provider(self, ext_type: str):
        for prov in self.providerMap.values():
            if prov.extType == ext_type:
                return prov

    def register_model(self, model):
        self.modelMap[model.uid] = t.cast(gws.IDatabaseModel, model)

    def model(self, model_uid):
        return self.modelMap.get(model_uid)

    def models(self):
        return list(self.modelMap.values())

    ##

    def table_uid(self, provider: gws.IDatabaseProvider, table_name: str) -> str:
        return provider.uid + ':' + provider.qualified_table_name(table_name)

    def registry_for_provider(self, provider):
        if not self.rt.registries.get(provider.uid):
            self.rt.registries[provider.uid] = sa.orm.registry()
        return self.rt.registries[provider.uid]

    def engine_for_provider(self, provider):
        if not self.rt.engines.get(provider.uid):
            self.rt.engines[provider.uid] = provider.engine()
        return self.rt.engines[provider.uid]

    def table_for_model(self, model) -> sa.Table:
        return self.rt.modelTable[model.uid]

    def class_for_model(self, model) -> type:
        return self.rt.modelClass[model.uid]

    def primary_keys_for_model(self, model) -> list[sa.Column]:
        return self.rt.primaryKeyColumns.get(model.uid, [])

    def table(self, provider: gws.IDatabaseProvider, table_name: str, columns: list[sa.Column] = None, **kwargs):
        tid = self.table_uid(provider, table_name)
        if tid in self.rt.tables and not columns:
            return self.rt.tables[tid]
        metadata = self.registry_for_provider(provider).metadata
        schema, name = provider.split_table_name(table_name)
        kwargs.setdefault('schema', schema)
        if columns:
            kwargs.setdefault('extend_existing', True)
            table = sa.Table(name, metadata, *columns, **kwargs)
        else:
            table = sa.Table(name, metadata, **kwargs)
        self.rt.tables[tid] = table
        return table

    def table_columns(self, table: sa.Table) -> list[sa.Column]:
        return t.cast(list[sa.Column], table.columns)

    def session(self, provider, **kwargs):
        if not self.rt.sessions.get(provider.uid):
            self.rt.sessions[provider.uid] = session_module.Object(provider)
        return self.rt.sessions[provider.uid]

    def close_sessions(self):
        for provider_uid, sess in self.rt.sessions.items():
            sess.saSession.close()
        self.rt.sessions = {}

    def autoload(self, sess, table_name):
        ins: sa.Inspector = sa.inspect(sess.saSession.connection())
        table = self.table(sess.provider, table_name)
        ins.reflect_table(table, include_columns=None)
        return table

    def describe(self, sess, table_name):
        tid = self.table_uid(sess.provider, table_name)
        if tid not in self.rt.describeCache:
            self.rt.describeCache[tid] = self._describe(sess, table_name)
        return self.rt.describeCache[tid]

    # https://www.psycopg.org/docs/usage.html#adaptation-of-python-values-to-sql-types

    SA_TO_ATTR = {
        'ARRAY': gws.AttributeType.strlist,
        'BIGINT': gws.AttributeType.int,
        'BIGSERIAL': gws.AttributeType.int,
        'BIT': gws.AttributeType.int,
        'BOOL': gws.AttributeType.bool,
        'BOOLEAN': gws.AttributeType.bool,
        'BYTEA': gws.AttributeType.bytes,
        'CHAR': gws.AttributeType.str,
        'CHARACTER VARYING': gws.AttributeType.str,
        'CHARACTER': gws.AttributeType.str,
        'DATE': gws.AttributeType.date,
        'DECIMAL': gws.AttributeType.float,
        'DOUBLE PRECISION': gws.AttributeType.float,
        'FLOAT4': gws.AttributeType.float,
        'FLOAT8': gws.AttributeType.float,
        'GEOMETRY': gws.AttributeType.geometry,
        'INT': gws.AttributeType.int,
        'INT2': gws.AttributeType.int,
        'INT4': gws.AttributeType.int,
        'INT8': gws.AttributeType.int,
        'INTEGER': gws.AttributeType.int,
        'MONEY': gws.AttributeType.float,
        'NUMERIC': gws.AttributeType.float,
        'REAL': gws.AttributeType.float,
        'SERIAL': gws.AttributeType.int,
        'SERIAL2': gws.AttributeType.int,
        'SERIAL4': gws.AttributeType.int,
        'SERIAL8': gws.AttributeType.int,
        'SMALLINT': gws.AttributeType.int,
        'SMALLSERIAL': gws.AttributeType.int,
        'TEXT': gws.AttributeType.str,
        'TIME': gws.AttributeType.time,
        'TIMESTAMP': gws.AttributeType.datetime,
        'TIMESTAMPTZ': gws.AttributeType.datetime,
        'TIMETZ': gws.AttributeType.time,
        'VARCHAR': gws.AttributeType.str,
    }
    SA_TO_GEOM = {
        'GEOMETRY': gws.GeometryType.geometry,
        'POINT': gws.GeometryType.point,
        'LINESTRING': gws.GeometryType.linestring,
        'POLYGON': gws.GeometryType.polygon,
        'MULTIPOINT': gws.GeometryType.multipoint,
        'MULTILINESTRING': gws.GeometryType.multilinestring,
        'MULTIPOLYGON': gws.GeometryType.multipolygon,
        'GEOMETRYCOLLECTION': gws.GeometryType.geometrycollection,
        'CURVE': gws.GeometryType.curve,
    }

    def _describe(self, sess: session_module.Object, table_name):
        ins: sa.Inspector = sa.inspect(sess.saSession.connection())

        schema, name = sess.provider.split_table_name(table_name)
        if not ins.has_table(name, schema):
            return None

        desc = gws.DataSetDescription(
            columns={},
            keyNames=[],
            schema=schema,
            name=name,
            fullName=sess.provider.join_table_name(name, schema),
            relationships=[],
        )

        for c in ins.get_columns(name, schema=schema):
            col = gws.ColumnDescription(
                comment=c.get('comment'),
                default=c.get('default'),
                isAutoincrement=bool(c.get('autoincrement')),
                isNullable=bool(c.get('nullable')),
                isPrimaryKey=False,
                isForeignKey=False,
                name=c.get('name'),
                options=gws.merge({}, c.get('dialect_options'), identity=c.get('identity')),
            )

            typ = c.get('type')
            col.nativeType = str(typ).upper()

            gt = getattr(typ, 'geometry_type', None)
            if gt:
                col.type = gws.AttributeType.geometry
                col.geometryType = gt.lower()
                col.geometrySrid = getattr(typ, 'srid')
            else:
                col.type = self.SA_TO_ATTR.get(col.nativeType, gws.AttributeType.str)

            desc.columns[col.name] = col

        c = ins.get_pk_constraint(desc.name, desc.schema)
        for name in c['constrained_columns']:
            desc.columns[name].isPrimaryKey = True
            desc.keyNames.append(name)

        for c in ins.get_foreign_keys(desc.name, desc.schema):
            rel = gws.RelationshipDescription(
                name=c['referred_table'],
                schema=c['referred_schema'],
                fullName=sess.provider.join_table_name(c['referred_table'], c['referred_schema']),
                foreignKeys=list(c['constrained_columns']),
                referredKeys=list(c['referred_columns']),
            )
            desc.relationships.append(rel)
            for name in rel.foreignKeys:
                desc.columns[name].isForeignKey = True

        for col in desc.columns.values():
            if col.geometryType:
                desc.geometryName = col.name
                desc.geometryType = col.geometryType
                desc.geometrySrid = col.geometrySrid
                break

        return desc


class _SaOrmCreateData:
    tid_to_models = {}
    tid_to_name = {}
    tid_to_provider = {}
    tid_to_table = {}
    tid_to_class = {}

    model_uid_to_tid = {}


class _SaRuntime:
    def __init__(self, mgr: Object):
        self.m = mgr
        self.registries = {}
        self.engines = {}
        self.sessions = {}
        self.tables = {}
        self.modelTable = {}
        self.modelClass = {}
        self.primaryKeyColumns = {}
        self.describeCache: dict[str, gws.DataSetDescription] = {}

    def create_orm(self):
        for model_uid in self.m.modelMap:
            self.primaryKeyColumns[model_uid] = []
            self.modelTable[model_uid] = None
            self.modelClass[model_uid] = None

        oc = _SaOrmCreateData()

        self._enumerate_tables(oc)

        deps = self._compute_table_dependencies(oc)

        tids = self._top_sort(oc.tid_to_models, deps)

        for tid in tids:
            oc.tid_to_table[tid] = self._create_table(tid, oc)

        for tid in tids:
            oc.tid_to_class[tid] = self._create_class(tid, oc)

        for tid in tids:
            for model in oc.tid_to_models[tid]:
                self.modelTable[model.uid] = oc.tid_to_table[tid]
                self.modelClass[model.uid] = oc.tid_to_class[tid]

    def _enumerate_tables(self, oc: _SaOrmCreateData):
        for model in self.m.modelMap.values():
            table_name = getattr(model, 'tableName', None)
            if not table_name:
                continue

            provider = t.cast(gws.IDatabaseProvider, getattr(model, 'provider'))
            tid = self.m.table_uid(provider, table_name)

            oc.model_uid_to_tid[model.uid] = tid
            oc.tid_to_models.setdefault(tid, []).append(model)
            oc.tid_to_provider[tid] = provider
            oc.tid_to_name[tid] = table_name

    def _compute_table_dependencies(self, oc: _SaOrmCreateData):

        deps = {}

        for tid, models in oc.tid_to_models.items():
            for model in models:
                for f in model.fields:
                    for dep_model_uid in f.orm_depends_on():
                        if dep_model_uid not in self.m.modelMap:
                            raise gws.Error(f'field {model.uid}.{f.name} depends on non-existing model {dep_model_uid!r}')
                        dep_tid = oc.model_uid_to_tid[dep_model_uid]
                        deps.setdefault(tid, []).append(dep_tid)

        return deps

    def _create_table(self, tid, oc: _SaOrmCreateData):
        name_to_col = {}
        name_to_model = {}

        for model in oc.tid_to_models[tid]:
            for f in model.fields:
                for col in f.orm_columns():
                    name = col.name
                    if name not in name_to_col:
                        name_to_col[name] = col
                        name_to_model[name] = model
                    else:
                        self._check_column_conflicts(
                            model,
                            col,
                            name_to_model[name],
                            name_to_col[name]
                        )
                    if f.isPrimaryKey:
                        self.primaryKeyColumns[model.uid].append(col)

        provider = oc.tid_to_provider[tid]
        name = oc.tid_to_name[tid]
        return self.m.table(provider, name, list(name_to_col.values()))

    def _create_class(self, tid, oc: _SaOrmCreateData):
        cls = type('FeatureRecord_' + gws.to_uid(tid), (gws.FeatureRecord,), {})
        provider = oc.tid_to_provider[tid]
        table = oc.tid_to_table[tid]
        self.m.registry_for_provider(provider).map_imperatively(cls, table)
        return cls

    def _add_props(self, tid, oc: _SaOrmCreateData):
        props = {}

        for model in oc.tid_to_models[tid]:
            for f in model.fields:
                props.update(f.orm_properties())

        cls = oc.tid_to_class[tid]

        for k, v in props.items():
            sa.orm.add_mapped_attribute(cls, k, v)

    def _top_sort(self, model_uids, model_deps):
        res = []

        def visit(uid, stack):
            if uid in res:
                return
            if uid in stack:
                s = '->'.join(stack)
                raise gws.Error(f'circular model dependency: {s}->{uid}')
            for dep in model_deps.get(uid, []):
                visit(dep, stack + [uid])
            res.append(uid)

        for model_uid in model_uids:
            visit(model_uid, [])

        return res

    _CHECK_KEYS = [
        'autoincrement',
        'comment',
        'computed',
        'constraints',
        'default',
        'description',
        'dialect_options',
        'doc',
        'foreign_keys',
        'identity',
        'index',
        'info',
        'inherit_cache',
        'is_clause_element',
        'is_dml',
        'is_literal',
        'is_selectable',
        'key',
        'name',
        'nullable',
        'onpudate',
        'onupdate',
        'primary_key',
        'server_default',
        'server_onupdate',
        'stringify_dialect',
        'supports_execution',
        'system',
        'table',
        'type',
        'unique',
        'uses_inspection',
    ]

    def _check_column_conflicts(self, new_model, new_col, old_model, old_col):
        for k in self._CHECK_KEYS:
            new_val = getattr(new_col, k, None)
            old_val = getattr(old_col, k, None)
            if repr(old_val) != repr(new_val):
                gws.log.debug(
                    f'column conflict: '
                    + f'{new_model.uid}.{new_col.name} {k}={new_val!r}'
                    + ' <> '
                    + f'{old_model.uid}.{old_col.name} {k}={old_val!r}')
