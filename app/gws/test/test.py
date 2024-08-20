"""Test configurator and invoker.

This script runs on the host machine.

Its purpose is to create a docker compose file, start the compose
and invoke the test runner inside the GWS container (via ``gws test``).
"""

import os
import sys
import yaml
import json

APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../..')
sys.path.insert(0, APP_DIR)

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
    
    -d, --detach          - run docker compose in the background
    -c, --coverage        - produce a coverage report
    -o, --only <regex>    - only run filenames matching the pattern 
    -v, --verbose         - enable debug logging
        
Pytest options:
    See https://docs.pytest.org/latest/reference.html#command-line-flags

"""

OPTIONS = {}


def main(args):
    cmd = args.get(1)

    ini_paths = [APP_DIR + '/test.ini']
    custom_ini = args.get('ini') or gws.env.GWS_TEST_INI
    if custom_ini:
        ini_paths.append(custom_ini)
    cli.info(f'using configs: {ini_paths}')
    OPTIONS.update(inifile.from_paths(*ini_paths))

    OPTIONS.update(dict(
        arg_ini=custom_ini,
        arg_manifest=args.get('manifest'),
        arg_detach=args.get('d') or args.get('detach'),
        arg_coverage=args.get('c') or args.get('coverage'),
        arg_only=args.get('o') or args.get('only'),
        arg_verbose=args.get('v') or args.get('verbose'),
        arg_pytest=args.get('_rest'),
    ))

    OPTIONS['APP_DIR'] = APP_DIR

    p = OPTIONS.get('runner.base_dir') or gws.env.GWS_TEST_DIR
    if not os.path.isabs(p):
        p = os.path.realpath(os.path.join(APP_DIR, p))
    OPTIONS['BASE_DIR'] = p

    OPTIONS['runner.uid'] = int(OPTIONS.get('runner.uid') or os.getuid())
    OPTIONS['runner.gid'] = int(OPTIONS.get('runner.gid') or os.getgid())

    if cmd == 'go':
        OPTIONS['arg_coverage'] = True
        OPTIONS['arg_detach'] = True
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

    ensure_dir(f'{base}/config', clear=True)
    ensure_dir(f'{base}/data')
    ensure_dir(f'{base}/gws-var')
    ensure_dir(f'{base}/pytest_cache')
    ensure_dir(f'{base}/tmp', clear=True)

    manifest_text = '{}'
    if OPTIONS['arg_manifest']:
        manifest_text = read_file(OPTIONS['arg_manifest'])

    write_file(f'{base}/config/MANIFEST.json', manifest_text)
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
        cmd += f' --only ' + OPTIONS['arg_only']
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

    OPTIONS['runner.services'] = list(service_funcs)

    for s, fn in service_funcs.items():
        srv = fn()

        srv.setdefault('image', OPTIONS.get(f'service.{s}.image'))
        srv.setdefault('extra_hosts', []).append(f"{OPTIONS.get('runner.docker_host_name')}:host-gateway")

        std_vols = [
            f'{base}:{base}',
            f'{APP_DIR}:/gws-app',
            f'{base}/data:/data',
            f'{base}/gws-var:/gws-var',
        ]
        srv.setdefault('volumes', []).extend(std_vols)

        srv.setdefault('tmpfs', []).append('/tmp')
        srv.setdefault('stop_grace_period', '1s')

        srv['environment'] = make_env()

        service_configs[s] = srv

    cfg = {
        'networks': {
            'default': {
                'name': 'gws_test_network'
            }
        },
        'services': service_configs,
    }

    return yaml.dump(cfg)


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
        'run.source': '/gws-app/gws',
        'run.data_file': f'{base}/coverage.data',
        'html.directory': f'{base}/coverage'
    }
    return inifile.to_string(ini)


def make_env():
    env = {
        'PYTHONPATH': '/gws-app',
        'GWS_UID': OPTIONS.get('runner.uid'),
        'GWS_GID': OPTIONS.get('runner.gid'),
        'GWS_TIMEZONE': OPTIONS.get('service.gws.time_zone', 'UTC'),
        'POSTGRES_DB': OPTIONS.get('service.postgres.database'),
        'POSTGRES_PASSWORD': OPTIONS.get('service.postgres.password'),
        'POSTGRES_USER': OPTIONS.get('service.postgres.user'),
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
            f"{OPTIONS.get('service.gws.http_expose_port')}:80",
            f"{OPTIONS.get('service.gws.mpx_expose_port')}:5000",
        ],
    )


def service_qgis():
    return dict(
        container_name='c_qgis',
        command=f'/bin/sh /qgis-start.sh',
        ports=[
            f"{OPTIONS.get('service.qgis.expose_port')}:80",
        ],
    )


_POSTGRESQL_CONF = """
listen_addresses = '*'
max_wal_size = 1GB
min_wal_size = 80MB
log_timezone = 'Etc/UTC'
datestyle = 'iso, mdy'
timezone = 'Etc/UTC'
default_text_search_config = 'pg_catalog.english'

logging_collector = 0
log_line_prefix = '%t %c %a %r '
log_statement = 'all'
log_connections = 1
log_disconnections = 1
log_duration = 1
log_hostname = 0
"""

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

    ep = write_exec(f'{base}/config/postgres_entrypoint', _POSTGRESQL_ENTRYPOINT)
    cf = write_file(f'{base}/config/postgresql.conf', _POSTGRESQL_CONF)

    ensure_dir(f'{base}/postgres')

    return dict(
        container_name='c_postgres',
        entrypoint=ep,
        ports=[
            f"{OPTIONS.get('service.postgres.expose_port')}:5432",
        ],
        volumes=[
            f"{base}/postgres:/var/lib/postgresql/data",
            f"{cf}:/etc/postgresql/postgresql.conf",
        ]
    )


def service_mockserver():
    return dict(
        # NB use the gws image
        container_name='c_mockserver',
        image=OPTIONS.get('service.gws.image'),
        command=f'python3 /gws-app/gws/test/mockserver.py',
        ports=[
            f"{OPTIONS.get('service.mockserver.expose_port')}:80",
        ],
    )


##

def docker_compose_start(with_exec=False):
    dc = OPTIONS['BASE_DIR'] + '/config/docker-compose.yml'

    cmd = ['docker', 'compose', '--file', dc, 'up']
    if OPTIONS['arg_detach']:
        cmd.append('--detach')

    if with_exec:
        return os.execvp('docker', cmd)

    cli.run(cmd)


def docker_compose_stop():
    dc = OPTIONS['BASE_DIR'] + '/config/docker-compose.yml'
    try:
        cli.run(f'docker compose --file {dc} down')
    except:
        pass


def docker_exec(container, cmd):
    opts = OPTIONS.get('runner.docker_exec_options', '')
    uid = OPTIONS.get('runner.uid')
    gid = OPTIONS.get('runner.gid')

    cli.run(f'''
        docker exec 
        --user {uid}:{gid}
        --env PYTHONPYCACHEPREFIX=/tmp
        --env PYTHONDONTWRITEBYTECODE=1
        {opts} 
        {container} 
        {cmd}
    ''')


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def write_file(path, s):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(s)
    return path


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


##


if __name__ == '__main__':
    cli.main('test', main, USAGE)
