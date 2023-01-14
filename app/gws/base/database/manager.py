import gws
import gws.types as t

from . import core, sql


class Config(gws.Config):
    """Database configuration"""

    providers: t.List[gws.ext.config.db]
    """database providers"""


class _SaOrmBase:
    feature: gws.IFeature


class _SaRuntime:
    registries: t.Dict[str, sql.orm.registry]
    engines: t.Dict[str, sql.sa.engine.Engine]
    sessions: t.Dict[str, sql.Session]
    tables: t.Dict[str, sql.sa.Table]
    classes: t.Dict[str, type]
    pkColumns: t.Dict[str, t.List[sql.sa.Column]]
    descCache: t.Dict[str, gws.DataSetDescription]

    def __init__(self):
        self.registries = {}
        self.engines = {}
        self.sessions = {}
        self.tables = {}
        self.classes = {}
        self.pkColumns = {}
        self.descCache = {}


class Object(gws.Node, gws.IDatabaseManager):
    models: t.Dict[str, gws.IModel]
    providers: t.Dict[str, gws.IDatabaseProvider]
    rt: _SaRuntime

    def __getstate__(self):
        state = dict(vars(self))
        del state['rt']
        return state

    def configure(self):
        self.providers = {}
        self.models = {}
        self.rt = _SaRuntime()

        for cfg in self.var('providers', default=[]):
            cfg = gws.merge(cfg, _manager=self)
            prov = self.root.create_shared(gws.ext.object.db, cfg)
            self.providers[prov.uid] = prov

    def provider(self, uid, ext_type):
        prov = self.providers.get(uid)
        if prov and prov.extType == ext_type:
            return prov

    def first_provider(self, ext_type: str):
        for prov in self.providers.values():
            if prov.extType == ext_type:
                return prov

    def provider_for(self, obj: gws.INode, ext_type: str = None):
        db = obj.var('db')
        if not db and obj.var('_provider'):
            return obj.var('_provider')

        ext_type = ext_type or obj.extType

        if db:
            p = self.provider(db, ext_type)
            if not p:
                raise gws.Error(f'database provider {ext_type!r} {db!r} not found')
            return p

        p = self.first_provider(ext_type)
        if not p:
            raise gws.Error(f'database provider {ext_type!r} not found')
        return p

    def register_model(self, model):
        self.models[model.uid] = model

    def model(self, model_uid):
        return self.models.get(model_uid)

    def activate(self):
        self.activate_runtime()

    def activate_runtime(self):
        self.rt = _SaRuntime()

        for prov_uid in self.providers:
            self.rt.registries[prov_uid] = sql.orm.registry()

        model_to_tkey = {}

        for model_uid, mod in self.models.items():
            table_name = getattr(mod, 'tableName', None)
            if not table_name:
                continue
            provider = t.cast(gws.IDatabaseProvider, getattr(mod, 'provider'))
            model_to_tkey[model_uid] = provider.uid, provider.qualified_table_name(table_name)

        tkey_to_cols = {}
        tkey_to_prim = {}

        for model_uid, tkey in model_to_tkey.items():
            d = {}
            for f in self.models[model_uid].fields:
                if f.isPrimaryKey:
                    f.sa_columns(d)
            tkey_to_prim.setdefault(tkey, {}).update(d)
            tkey_to_cols.setdefault(tkey, {}).update(d)

        for model_uid, tkey in model_to_tkey.items():
            if tkey in tkey_to_prim:
                self.rt.pkColumns[model_uid] = list(tkey_to_prim[tkey].values())

        for model_uid, tkey in model_to_tkey.items():
            d = {}
            for f in self.models[model_uid].fields:
                if not f.isPrimaryKey:
                    f.sa_columns(d)
            tkey_to_cols.setdefault(tkey, {}).update(d)

        tkey_to_table = {}
        tkey_to_class = {}

        for tkey, cols in tkey_to_cols.items():
            provider_uid, table_name = tkey
            provider = self.providers[provider_uid]
            tab = self.sa_make_table(provider, table_name, cols.values())
            cls = type('_SA_' + provider_uid + '_' + gws.to_uid(table_name), (_SaOrmBase,), {})
            self.rt.registries[provider_uid].map_imperatively(cls, tab)
            tkey_to_table[tkey] = tab
            tkey_to_class[tkey] = cls

        for model_uid, tkey in model_to_tkey.items():
            self.rt.tables[model_uid] = tkey_to_table[tkey]
            self.rt.classes[model_uid] = tkey_to_class[tkey]

        tkey_to_props = {}

        for model_uid, tkey in model_to_tkey.items():
            d = {}
            for f in self.models[model_uid].fields:
                f.sa_properties(d)
            tkey_to_props.setdefault(tkey, {}).update(d)

        for tkey, props in tkey_to_props.items():
            for k, v in props.items():
                getattr(tkey_to_class[tkey], '__mapper__').add_property(k, v)

    def sa_table(self, model) -> sql.sa.Table:
        return self.rt.tables[model.uid]

    def sa_class(self, model) -> type:
        return self.rt.classes[model.uid]

    def sa_pk_columns(self, model) -> t.List[sql.sa.Column]:
        return self.rt.pkColumns.get(model.uid, [])

    def sa_make_table(self, provider: gws.IDatabaseProvider, name: str, columns: t.List[sql.sa.Column], **kwargs):
        metadata = self.rt.registries[provider.uid].metadata

        if '.' in name:
            schema, name = name.split('.')
            kwargs.setdefault('schema', schema)

        return sql.sa.Table(name, metadata, *columns, **kwargs)

    def sa_make_engine(self, drivername: str, options, **kwargs):
        url = sql.sa.engine.URL.create(
            drivername,
            username=options.get('username'),
            password=options.get('password'),
            host=options.get('host'),
            port=options.get('port'),
            database=options.get('database'),
        )
        return sql.sa.create_engine(url, **kwargs)

    def session(self, provider, **kwargs):
        if not self.rt.engines.get(provider.uid):
            gws.log.debug(f'db: create engine for {provider.uid!r}')
            self.rt.engines[provider.uid] = provider.sa_engine()

        if not self.rt.sessions.get(provider.uid):
            gws.log.debug(f'db: create session for {provider.uid!r}')
            self.rt.sessions[provider.uid] = sql.Session(
                sql.orm.Session(self.rt.engines[provider.uid], future=True, **kwargs))

        return self.rt.sessions[provider.uid]

    def close_sessions(self):
        for provider_uid, sess in self.rt.sessions.items():
            gws.log.debug(f'db: close session for {provider_uid!r}')
            sess.saSession.close()
        self.rt.sessions = {}

    def describe_table(self, provider: gws.IDatabaseProvider, table_name: str) -> t.Optional[gws.DataSetDescription]:
        cache_key = gws.join_uid(provider.uid, table_name)
        if cache_key not in self.rt.descCache:
            self.rt.descCache[cache_key] = self._describe_table_impl(provider, table_name)
        return self.rt.descCache[cache_key]

    def _describe_table_impl(self, provider, table_name):
        ins = sql.sa.inspect(provider.sa_engine())

        desc = gws.DataSetDescription(
            columns={},
            keyNames=[],
        )

        if '.' in table_name:
            s, _, n = table_name.partition('.')
            desc.name = n
            desc.schema = s
            desc.fullName = s + '.' + n
        else:
            desc.name = table_name
            desc.schema = ''
            desc.fullName = table_name

        try:
            ins_columns = ins.get_columns(desc.name, desc.schema)
        except sql.exc.NoSuchTableError:
            return None

        for c in ins_columns:
            col = gws.ColumnDescription(
                comment=c.get('comment', ''),
                default=c.get('default'),
                isAutoincrement=c.get('autoincrement', False),
                isNullable=c.get('nullable', False),
                isPrimaryKey=False,
                name=c.get('name'),
                options=c.get('dialect_options', {}),
            )

            typ = c.get('type')
            col.nativeType = str(typ).upper()

            gt = getattr(typ, 'geometry_type', None)
            if gt:
                col.type = gws.AttributeType.geometry
                col.geometryType = gt.lower()
                col.geometrySrid = getattr(typ, 'srid')
            else:
                col.type = sql.SA_TO_ATTR.get(col.nativeType, gws.AttributeType.str)

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
