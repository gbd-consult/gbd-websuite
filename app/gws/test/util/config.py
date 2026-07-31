"""Configuration and the root object."""

import gws
import gws.config
import gws.lib.vendor.slon
import gws.spec.runtime

from . import auth, options


def _config_defaults():
    return f"""
        database.providers+ {{
            uid "GWS_TEST_POSTGRES_PROVIDER"
            type "postgres"
            host     {options.option('service.postgres.host')!r}
            port     {int(options.option('service.postgres.port'))}
            username {options.option('service.postgres.user')!r}
            password {options.option('service.postgres.password')!r}
            database {options.option('service.postgres.database')!r}
            schemaCacheLifeTime 0
        }}
    """


def _to_data(x):
    if isinstance(x, gws.Data):
        for k, v in vars(x).items():
            setattr(x, k, _to_data(v))
        return x
    if isinstance(x, dict):
        d = gws.Data()
        for k, v in x.items():
            setattr(d, k, _to_data(v))
        return d
    if isinstance(x, list):
        return [_to_data(y) for y in x]
    if isinstance(x, tuple):
        return tuple(_to_data(y) for y in x)
    return x


_SPEC_RUNTIME = None


def gws_specs() -> gws.SpecRuntime:
    global _SPEC_RUNTIME

    if _SPEC_RUNTIME is None:
        base = options.option('BASE_DIR')
        _SPEC_RUNTIME = gws.spec.runtime.create(f'{base}/config/MANIFEST.json', read_cache=False, write_cache=False)

    return _SPEC_RUNTIME


def gws_root(cfg: str = '', specs: gws.SpecRuntime = None, activate=True, defaults=True, **vars):
    cfg = cfg or ''
    if defaults:
        cfg = _config_defaults() + '\n' + cfg

    cfg = f'server.log.level {gws.log.get_level()}\n' + cfg

    for k, v in vars.items():
        cfg = cfg.replace('{' + k + '}', str(v))

    parsed_config = _to_data(gws.lib.vendor.slon.parse(cfg, as_object=True))
    specs = auth.register(specs or gws_specs())
    root = gws.config.initialize(specs, gws.Config(parsed_config))

    if root.configErrors:
        for err in root.configErrors:
            gws.log.error(f'CONFIGURATION ERROR: {err}')
        raise gws.ConfigurationError('config failed')

    if not activate:
        return root

    root = gws.config.activate(root)
    return root
