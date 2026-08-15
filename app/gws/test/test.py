"""Test configurator and invoker.

This script runs on the host machine.

Its purpose is to create a docker compose file, start the compose
and invoke the test runner inside the GWS container (via ``gws test``).
"""

import os
import re
import sys
import yaml
import json

LOCAL_APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../..')
sys.path.insert(0, LOCAL_APP_DIR)

import gws
import gws.lib.cli as cli
import gws.lib.inifile as inifile

USAGE = """
GWS test runner
~~~~~~~~~~~~~~~

    python3 test.py <command> <options> - <pytest options>

Commands:

    test.py go
        - start the test environment, run tests and stop

    test.py start
        - start the compose test environment

    test.py stop
        - stop the compose test environment
        
    test.py run
        - run tests in a started environment
        
Options:
    --ini <path>          - path to the local 'ini' file (can also be passed in the GWS_TEST_INI env var)
    --manifest <manifest> - path to MANIFEST.json
    
    -b, --batch           - run tests in batch mode (no interactive prompts)
    -c, --coverage        - produce a coverage report
    -d, --detach          - run docker compose in the background
    -l, --local           - mount the local copy of the application in the test container   
    -o, --only <regex>    - only run filenames matching the pattern 
    -k, --keyword <expr>  - only run tests matching the pytest expression
    -v, --verbose         - enable debug logging
        
Pytest options:
    See https://docs.pytest.org/en/latest/reference/reference.html#command-line-flags

"""

OPTIONS = {}

CONTAINER_DATA_DIR = '/data'
DEFAULT_SOURCE_DIR = '/gws-app/gws'
COVERAGE_OMIT = ['*/___*', '*/vendor/*', '*_test.py', '*/conftest.py', '*/test/*']


def main(args):
    cmd = args.get(1)

    ini_paths = [LOCAL_APP_DIR + '/test.ini']
    custom_ini = args.get('ini') or gws.env.GWS_TEST_INI
    if custom_ini:
        ini_paths.append(custom_ini)
    cli.info(f'using configs: {ini_paths}')
    for path in ini_paths:
        OPTIONS.update(load_ini(path))

    OPTIONS.update(
        dict(
            arg_ini=custom_ini,
            arg_pytest=args.get('_rest'),
            arg_batch=args.get('b') or args.get('batch'),
            arg_coverage=args.get('c') or args.get('coverage'),
            arg_detach=args.get('d') or args.get('detach'),
            arg_local=args.get('l') or args.get('local'),
            arg_manifest=args.get('manifest'),
            arg_only=args.get('o') or args.get('only'),
            arg_keyword=args.get('k') or args.get('keyword'),
            arg_verbose=args.get('v') or args.get('verbose'),
        )
    )

    OPTIONS['LOCAL_APP_DIR'] = LOCAL_APP_DIR
    OPTIONS['HOST_OS'] = sys.platform

    p = OPTIONS.get('runner.base_dir') or abs_path(gws.env.GWS_TEST_DIR, LOCAL_APP_DIR)
    if not p:
        raise ValueError('GWS_TEST_DIR not set')
    OPTIONS['BASE_DIR'] = p

    OPTIONS['runner.uid'] = int(OPTIONS.get('runner.uid') or os.getuid())
    OPTIONS['runner.gid'] = int(OPTIONS.get('runner.gid') or os.getgid())

    OPTIONS['runner.data_dir'] = OPTIONS.get('runner.data_dir') or OPTIONS['BASE_DIR'] + CONTAINER_DATA_DIR

    if cmd == 'go':
        OPTIONS['arg_coverage'] = True
        OPTIONS['arg_detach'] = True
        OPTIONS['arg_batch'] = True
        docker_compose_stop()
        configure()
        docker_compose_start()
        run()
        docker_compose_stop()
        return 0

    if cmd == 'start':
        docker_compose_stop()
        configure()
        docker_compose_start(with_exec=True)
        return 0

    if cmd == 'stop':
        docker_compose_stop()
        return 0

    if cmd == 'run':
        run()
        return 0

    cli.fatal('invalid arguments, try test.py -h for help')


##


def configure():
    base = OPTIONS['BASE_DIR']

    data_dir = OPTIONS['runner.data_dir']

    ensure_dir(f'{base}/config', clear=True)
    if data_dir.startswith(base + '/'):
        ensure_dir(data_dir)
    elif not os.path.isdir(data_dir):
        cli.fatal(f'data dir {data_dir!r} not found')
    ensure_dir(f'{base}/gws-var')
    ensure_dir(f'{base}/pytest_cache')

    # extra volumes in the base dir are ours to create
    for vol in OPTIONS['runner.extra_volumes']:
        src = vol.split(':')[0]
        if src.startswith(base + '/'):
            ensure_dir(src)

    write_file(f'{base}/config/MANIFEST.json', make_manifest_text())
    write_file(f'{base}/config/docker-compose.yml', make_docker_compose_yml())
    write_file(f'{base}/config/pg_service.conf', make_pg_service_conf())
    write_file(f'{base}/config/pytest.ini', make_pytest_ini())
    write_file(f'{base}/config/coverage.ini', make_coverage_ini())
    write_file(f'{base}/config/OPTIONS.json', json.dumps(OPTIONS, indent=4))

    cli.info(f'tests configured in {base!r}')


def run():
    base = OPTIONS['BASE_DIR']
    coverage_ini = f'{base}/config/coverage.ini'

    cmd = ''

    if OPTIONS['arg_coverage']:
        cmd += f'coverage run --rcfile={coverage_ini}'
    else:
        cmd += 'python3'

    cmd += f' /gws-app/gws/test/container_runner.py --base {base}'

    if OPTIONS['arg_only']:
        cmd += f' --only "{OPTIONS["arg_only"]}"'
    if OPTIONS['arg_keyword']:
        cmd += f' --keyword "{OPTIONS["arg_keyword"]}"'
    if OPTIONS['arg_verbose']:
        cmd += ' --verbose '
    if OPTIONS['arg_pytest']:
        cmd += ' - ' + ' '.join(OPTIONS['arg_pytest'])

    docker_exec('c_gws', cmd)

    if OPTIONS['arg_coverage']:
        ensure_dir(f'{base}/coverage', clear=True)
        docker_exec('c_gws', f'coverage html --rcfile={coverage_ini}')
        docker_exec('c_gws', f'coverage report --rcfile={coverage_ini} --sort=cover > {base}/coverage/report.txt')


##


def make_docker_compose_yml():
    base = OPTIONS['BASE_DIR']

    service_configs = {}

    service_funcs = {}
    for k, v in globals().items():
        if k.startswith('service_'):
            service_funcs[k.split('_')[1]] = v

    wanted = OPTIONS['runner.services'] or list(service_funcs)
    for s in wanted:
        if s not in service_funcs:
            cli.fatal(f'unknown service {s!r}')
    service_funcs = {s: service_funcs[s] for s in wanted}

    OPTIONS['runner.services'] = list(service_funcs)

    std_env = make_std_env()

    for s, fn in service_funcs.items():
        srv = fn()

        srv.setdefault('image', OPTIONS.get(f'service.{s}.image'))
        srv.setdefault('extra_hosts', []).append(f'{OPTIONS.get("runner.docker_host_name")}:host-gateway')

        std_vols = [
            f'{base}:{base}',
            f'{OPTIONS["runner.data_dir"]}:{CONTAINER_DATA_DIR}',
            f'{base}/gws-var:/gws-var',
        ]
        if OPTIONS['arg_local']:
            std_vols.append(f'{LOCAL_APP_DIR}:/gws-app')
        std_vols.extend(OPTIONS['runner.extra_volumes'])

        srv.setdefault('volumes', []).extend(std_vols)

        srv.setdefault('tmpfs', []).append('/tmp')
        srv.setdefault('stop_grace_period', '1s')

        std_env.update(srv.get('environment', {}))
        service_configs[s] = srv

    for srv in service_configs.values():
        srv['environment'] = std_env

    cfg = {
        'networks': {'default': {'name': 'gws_test_network'}},
        'services': service_configs,
    }

    return yaml.dump(cfg)


def make_manifest_text():
    path = OPTIONS['arg_manifest']
    if not path:
        return '{}'

    js = json.loads(read_file(path))
    src_dir = os.path.dirname(os.path.abspath(path))

    for p in js.get('plugins', []):
        p['path'] = container_path(p['path'], src_dir)
    if js.get('tsConfig'):
        js['tsConfig'] = container_path(js['tsConfig'], src_dir)

    return json.dumps(js, indent=4)


def container_path(path, src_dir):
    if os.path.isabs(path):
        return path

    abs_path = os.path.abspath(os.path.join(src_dir, path))
    data_dir = OPTIONS['runner.data_dir']

    if abs_path == data_dir:
        return CONTAINER_DATA_DIR
    if abs_path.startswith(data_dir + '/'):
        return CONTAINER_DATA_DIR + abs_path[len(data_dir):]

    return abs_path


def make_pg_service_conf():
    name = OPTIONS.get('service.postgres.name')
    ini = {
        f'{name}.host': OPTIONS.get('service.postgres.host'),
        f'{name}.port': OPTIONS.get('service.postgres.port'),
        f'{name}.user': OPTIONS.get('service.postgres.user'),
        f'{name}.password': OPTIONS.get('service.postgres.password'),
        f'{name}.dbname': OPTIONS.get('service.postgres.database'),
    }
    return inifile.to_string(ini)


def make_pytest_ini():
    # https://docs.pytest.org/en/7.1.x/reference/reference.html#ini-OPTIONS-ref

    base = OPTIONS['BASE_DIR']
    ini = {}
    for k, v in OPTIONS.items():
        if k.startswith('pytest.'):
            ini[k] = v
    ini['pytest.cache_dir'] = f'{base}/pytest_cache'
    return inifile.to_string(ini)


def make_coverage_ini():
    # https://coverage.readthedocs.io/en/7.5.3/config.html

    base = OPTIONS['BASE_DIR']
    ini = {
        'run.source': ','.join(OPTIONS['runner.sources'] or [DEFAULT_SOURCE_DIR]),
        'run.omit': ','.join(COVERAGE_OMIT),
        'run.data_file': f'{base}/coverage.data',
        'html.directory': f'{base}/coverage',
    }
    return inifile.to_string(ini)


def make_std_env():
    base = OPTIONS['BASE_DIR']

    env = {
        'PYTHONPATH': '/gws-app',
        'PYTHONPYCACHEPREFIX': '/tmp',
        'PYTHONDONTWRITEBYTECODE': '1',
        'GWS_UID': OPTIONS.get('runner.uid'),
        'GWS_GID': OPTIONS.get('runner.gid'),
        'GWS_TIMEZONE': OPTIONS.get('service.gws.time_zone', 'Etc/UTC'),
        'PGSERVICEFILE': f'{base}/config/pg_service.conf',
    }

    for k, v in OPTIONS.items():
        sec, _, name = k.partition('.')
        if sec == 'environment':
            env[name] = v

    return env


##

_GWS_ENTRYPOINT = """
#!/usr/bin/env bash

groupadd --gid $GWS_GID g_$GWS_GID
useradd  --create-home --uid $GWS_UID --gid $GWS_GID u_$GWS_UID

ln -fs /usr/share/zoneinfo/$GWS_TIMEZONE /etc/localtime

sleep infinity
"""


def service_gws():
    base = OPTIONS['BASE_DIR']

    ep = write_exec(f'{base}/config/gws_entrypoint', _GWS_ENTRYPOINT)

    return dict(
        container_name='c_gws',
        entrypoint=ep,
        ports=[
            f'{OPTIONS.get("service.gws.http_expose_port")}:80',
            f'{OPTIONS.get("service.gws.mpx_expose_port")}:5000',
        ],
    )


def service_qgis():
    return dict(
        container_name='c_qgis',
        command=f'/bin/sh /qgis-start.sh',
        ports=[
            f'{OPTIONS.get("service.qgis.expose_port")}:80',
        ],
    )


_POSTGRESQL_ENTRYPOINT = """
#!/usr/bin/env bash

# delete existing and create our own postgres user
groupdel -f postgres
userdel -f postgres
groupadd --gid $GWS_GID postgres
useradd --create-home --uid $GWS_UID --gid $GWS_GID postgres

# invoke the original postgres entry point
docker-entrypoint.sh postgres --config_file=/etc/postgresql/postgresql.conf
"""


def service_postgres():
    # https://github.com/docker-library/docs/blob/master/postgres/README.md
    # https://github.com/postgis/docker-postgis

    # the entrypoint business is because
    # - 'postgres' uid should match host uid (or whatever is configured in test.ini)
    # - we need a custom config file

    base = OPTIONS['BASE_DIR']
    tz = OPTIONS.get('service.gws.time_zone', 'Etc/UTC')

    conf = f"""
        listen_addresses = '*'
        max_wal_size = 1GB
        min_wal_size = 80MB
        log_timezone = '{tz}'
        timezone = '{tz}'
        datestyle = 'iso, mdy'
        default_text_search_config = 'pg_catalog.english'

        logging_collector = 0
        log_line_prefix = '%t %c %a %r '
        log_statement = 'all'
        log_connections = 1
        log_disconnections = 1
        log_duration = 1
        log_hostname = 0
    """

    ep = write_exec(f'{base}/config/postgres_entrypoint', _POSTGRESQL_ENTRYPOINT)
    cf = write_file(f'{base}/config/postgresql.conf', dedent(conf))

    ensure_dir(f'{base}/postgres')

    return dict(
        container_name='c_postgres',
        entrypoint=ep,
        ports=[
            f'{OPTIONS.get("service.postgres.expose_port")}:5432',
        ],
        environment={
            'POSTGRES_DB': OPTIONS.get('service.postgres.database'),
            'POSTGRES_PASSWORD': OPTIONS.get('service.postgres.password'),
            'POSTGRES_USER': OPTIONS.get('service.postgres.user'),
        },
        volumes=[
            f'{base}/postgres:/var/lib/postgresql/data',
            f'{cf}:/etc/postgresql/postgresql.conf',
        ],
    )


def service_mockserver():
    return dict(
        # NB use the gws image
        container_name='c_mockserver',
        image=OPTIONS.get('service.gws.image'),
        # NB 'entrypoint', not 'command', because the image can define its own entrypoint
        entrypoint=f'python3 /gws-app/gws/test/mockserver.py',
        ports=[
            f'{OPTIONS.get("service.mockserver.expose_port")}:80',
        ],
    )


def service_ldap():
    base = OPTIONS['BASE_DIR']
    src = f'{LOCAL_APP_DIR}/gws/plugin/auth_provider/ldap/_test/'
    dst = f'{base}/config/ldap'
    ensure_dir(f'{dst}/custom')
    ensure_dir(f'{dst}/certs')
    
    copy_file(f'{src}/test.ldif', f'{dst}/custom/test.ldif')

    with open(f'{src}/certs.yaml') as fp:
        certs = yaml.safe_load(fp)
        for name, content in certs.items():
            write_file(f'{dst}/certs/{name}', content)
        os.chmod(f'{dst}/certs/ldap.key', 0o600)

    return dict(
        # NB: no _ in the host name
        container_name='cldap',
        environment={
            'LDAP_ORGANISATION': OPTIONS.get('service.ldap.organisation'),
            'LDAP_DOMAIN': 'example.com',
            'LDAP_ADMIN_PASSWORD': OPTIONS.get('service.ldap.password'),
            'LDAP_CONFIG_PASSWORD': OPTIONS.get('service.ldap.password'),
            'LDAP_TLS': 'true',
        },
        command='--copy-service --loglevel info',
        volumes=[
            f'{dst}/custom:/container/service/slapd/assets/config/bootstrap/ldif/custom',
            f'{dst}/certs:/container/service/slapd/assets/certs',
        ],
    )


##


def docker_compose_start(with_exec=False):
    cmd = [
        'docker',
        'compose',
        '--file',
        OPTIONS['BASE_DIR'] + '/config/docker-compose.yml',
        'up',
    ]
    if OPTIONS['arg_detach']:
        cmd.append('--detach')

    if with_exec:
        return os.execvp('docker', cmd)

    cli.run(cmd)


def docker_compose_stop():
    cmd = [
        'docker',
        'compose',
        '--file',
        OPTIONS['BASE_DIR'] + '/config/docker-compose.yml',
        'down',
    ]

    try:
        cli.run(cmd)
    except:
        pass


def docker_exec(container, cmd):
    uid = OPTIONS.get('runner.uid')
    gid = OPTIONS.get('runner.gid')

    it = '' if OPTIONS['arg_batch'] else '-it'

    cli.run(f"""
        docker exec 
        --user {uid}:{gid}
        --env PYTHONPYCACHEPREFIX=/tmp
        --env PYTHONDONTWRITEBYTECODE=1
        {it} 
        {container} 
        {cmd}
    """)


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def write_file(path, s):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(s)
    return path


def copy_file(src, dst):
    with open(src, 'rb') as fp:
        buf = fp.read()
    with open(dst, 'wb') as fp:
        fp.write(buf)


def write_exec(path, s):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(s.strip() + '\n')
    os.chmod(path, 0o777)
    return path


def ensure_dir(path, clear=False):
    def _clear(d):
        for de in os.scandir(d):
            if de.is_dir():
                _clear(de.path)
                os.rmdir(de.path)
            else:
                os.unlink(de.path)

    os.makedirs(path, exist_ok=True)
    if clear:
        _clear(path)


def load_ini(path):
    ini_dir = os.path.dirname(os.path.abspath(path))
    res = {}

    for key, val in inifile.from_paths_flat(path).items():
        if key.endswith('_dir'):
            val = abs_path(val, ini_dir) if val else ''
        elif key == 'runner.extra_volumes':
            val = [abs_volume(s, ini_dir) for s in split_list(val)]
        elif key in ('runner.services', 'runner.sources'):
            val = split_list(val)
        res[key] = val

    return res


def abs_path(path, base_dir):
    if not path:
        return ''
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.realpath(path)


def abs_volume(vol, base_dir):
    src, _, dst = vol.partition(':')
    return abs_path(src, base_dir) + ':' + dst


def split_list(val):
    if not val:
        return []
    if isinstance(val, list):
        return val
    return [s.strip() for s in re.split(r'[,\n]', val) if s.strip()]


def dedent(text):
    lines = text.split('\n')
    ind = 100_000
    for ln in lines:
        n = len(ln.lstrip())
        if n > 0:
            ind = min(ind, len(ln) - n)
    return '\n'.join(ln[ind:] for ln in lines)


##


if __name__ == '__main__':
    cli.main('test', main, USAGE)
