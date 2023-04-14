"""Core database utilities."""

import gws
import gws.lib.sa as sa

import gws.types as t

from . import session as session_module


class Config(gws.Config):
    """Database configuration"""

    providers: list[gws.ext.config.databaseProvider]
    """database providers"""


class _SaRuntime:
    def __init__(self):
        self.registries = {}
        self.engines = {}
        self.sessions = {}
        self.tables = {}
        self.modelTables = {}
        self.modelClasses = {}
        self.pkColumns = {}
        self.descCache = {}


class Object(gws.Node, gws.IDatabaseManager):
    models: dict[str, gws.IModel]
    providers: dict[str, gws.IDatabaseProvider]
    rt: _SaRuntime

    def __getstate__(self):
        state = dict(vars(self))
        del state['rt']
        return state

    def configure(self):
        self.providers = {}
        self.models = {}
        self.rt = _SaRuntime()

        for cfg in self.cfg('providers', default=[]):
            prov = self.root.create_shared(gws.ext.object.databaseProvider, cfg, _defaultManager=self)
            self.providers[prov.uid] = prov

    def post_configure(self):
        self.close_sessions()

    def provider(self, uid, ext_type):
        prov = self.providers.get(uid)
        if prov and prov.extType == ext_type:
            return prov

    def first_provider(self, ext_type: str):
        for prov in self.providers.values():
            if prov.extType == ext_type:
                return prov

    def register_model(self, model):
        self.models[model.uid] = model

    def model(self, model_uid):
        return self.models.get(model_uid)

    def activate(self):
        self.activate_runtime()

    def activate_runtime(self):
        self.rt = _SaRuntime()

        model_to_tuid = {}

        for model_uid, mod in self.models.items():
            table_name = getattr(mod, 'tableName', None)
            if not table_name:
                continue
            provider = t.cast(gws.IDatabaseProvider, getattr(mod, 'provider'))
            model_to_tuid[model_uid] = self.table_uid(provider, table_name)

        tuid_to_cols = {}
        tuid_to_prim = {}

        for model_uid, tuid in model_to_tuid.items():
            d = {}
            for f in self.models[model_uid].fields:
                if f.isPrimaryKey:
                    for col in f.columns():
                        d[col.name] = col
            tuid_to_prim.setdefault(tuid, {}).update(d)
            tuid_to_cols.setdefault(tuid, {}).update(d)

        for model_uid, tuid in model_to_tuid.items():
            if tuid in tuid_to_prim:
                self.rt.pkColumns[model_uid] = list(tuid_to_prim[tuid].values())

        for model_uid, tuid in model_to_tuid.items():
            d = {}
            for f in self.models[model_uid].fields:
                if not f.isPrimaryKey:
                    for col in f.columns():
                        d[col.name] = col
            tuid_to_cols.setdefault(tuid, {}).update(d)

        tuid_to_table = {}
        tuid_to_class = {}

        for tuid, cols in tuid_to_cols.items():
            provider_uid, table_name = tuid
            provider = self.providers[provider_uid]
            tab = self.table(provider, table_name, cols.values())
            cls = type('FeatureRecord_' + provider_uid + '_' + gws.to_uid(table_name), (gws.FeatureRecord,), {})
            self.registry_for_provider(provider).map_imperatively(cls, tab)
            tuid_to_table[tuid] = tab
            tuid_to_class[tuid] = cls

        for model_uid, tuid in model_to_tuid.items():
            self.rt.modelTables[model_uid] = tuid_to_table[tuid]
            self.rt.modelClasses[model_uid] = tuid_to_class[tuid]

        tuid_to_props = {}

        for model_uid, tuid in model_to_tuid.items():
            d = {}
            for f in self.models[model_uid].fields:
                d.update(f.orm_properties())
            tuid_to_props.setdefault(tuid, {}).update(d)

        for tuid, props in tuid_to_props.items():
            for k, v in props.items():
                getattr(tuid_to_class[tuid], '__mapper__').add_property(k, v)

    def table_uid(self, provider: gws.IDatabaseProvider, table_name: str):
        return provider.uid, provider.qualified_table_name(table_name)

    def registry_for_provider(self, provider):
        if not self.rt.registries.get(provider.uid):
            self.rt.registries[provider.uid] = sa.orm.registry()
        return self.rt.registries[provider.uid]

    def engine_for_provider(self, provider):
        if not self.rt.engines.get(provider.uid):
            self.rt.engines[provider.uid] = provider.engine()
        return self.rt.engines[provider.uid]

    def table_for_model(self, model) -> sa.Table:
        return self.rt.modelTables[model.uid]

    def class_for_model(self, model) -> type:
        return self.rt.modelClasses[model.uid]

    def pkeys_for_model(self, model) -> list[sa.Column]:
        return self.rt.pkColumns.get(model.uid, [])

    def table(self, provider: gws.IDatabaseProvider, table_name: str, columns: list[sa.Column] = None, **kwargs):
        tuid = self.table_uid(provider, table_name)
        if tuid in self.rt.tables and not columns:
            return self.rt.tables[tuid]
        metadata = self.registry_for_provider(provider).metadata
        schema, name = provider.parse_table_name(table_name)
        kwargs.setdefault('schema', schema)
        if columns:
            kwargs.setdefault('extend_existing', True)
            tab = sa.Table(name, metadata, *columns, **kwargs)
        else:
            tab = sa.Table(name, metadata, **kwargs)
        self.rt.tables[tuid] = tab
        return tab

    def make_engine(self, drivername: str, options, **kwargs):
        url = sa.URL.create(
            drivername,
            username=options.get('username'),
            password=options.get('password'),
            host=options.get('host'),
            port=options.get('port'),
            database=options.get('database'),
        )
        kwargs.setdefault('poolclass', sa.NullPool)
        kwargs.setdefault('pool_pre_ping', True)
        return sa.create_engine(url, **kwargs)

    def session(self, provider, **kwargs):
        if not self.rt.sessions.get(provider.uid):
            gws.log.debug(f'db: create session for {provider.uid!r}')
            self.rt.sessions[provider.uid] = session_module.Object(provider)

        return self.rt.sessions[provider.uid]

    def close_sessions(self):
        for provider_uid, sess in self.rt.sessions.items():
            gws.log.debug(f'db: close session for {provider_uid!r}')
            sess.saSession.close()
        self.rt.sessions = {}

    def autoload(self, sess, table_name):
        return self._load_and_describe(sess, table_name, True)

    def describe(self, sess, table_name):
        tuid = self.table_uid(sess.provider, table_name)
        if tuid not in self.rt.descCache:
            self.rt.descCache[tuid] = self._load_and_describe(sess, table_name, False)
        return self.rt.descCache[tuid]

    # http://initd.org/psycopg/docs/usage.html?highlight=smallint#adaptation-of-python-values-to-sql-types

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

    def _load_and_describe(self, sess, table_name, load_only):
        ins = sa.inspect(sess.saSession.connection())

        schema, name = sess.provider.parse_table_name(table_name)
        if not ins.has_table(name, schema):
            return None

        tab = self.table(sess.provider, table_name)
        ins.reflect_table(tab, include_columns=None)

        if load_only:
            return tab

        desc = gws.DataSetDescription(
            columns={},
            keyNames=[],
            schema=tab.schema,
            name=tab.name,
            qname=tab.schema + '.' + tab.name,
        )

        for c in t.cast(t.Iterable, tab.columns):
            col = gws.ColumnDescription(
                comment=c.comment,
                default=c.default,
                isAutoincrement=c.autoincrement,
                isNullable=c.nullable,
                isPrimaryKey=False,
                name=c.name,
                options=c.dialect_options
            )

            typ = c.type
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
            rel = c['referred_schema'] + '.' + c['referred_table']
            for name in c['constrained_columns']:
                desc.columns[name].relation = rel

        for col in desc.columns.values():
            if col.geometryType:
                desc.geometryName = col.name
                desc.geometryType = col.geometryType
                desc.geometrySrid = col.geometrySrid
                break

        return desc
