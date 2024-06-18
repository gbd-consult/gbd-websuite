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

import gws.lib.cli as cli
import gws.lib.inifile as inifile

USAGE = """
GWS test runner
~~~~~~~~~~~~~~~

    python3 test.py <command> <options>

Commands:

    test.py go [--ini <path>] [--manifest <manifest>]
        - start the test environment, run tests and stop

    test.py start [--ini <path>] [--manifest <manifest>] [-d | --detach] 
        - start the compose test environment

    test.py stop
        - stop the compose test environment
        
    test.py run [-c] [-o <regex>] [-v] - [pytest options]  
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

    ini_path = args.get('ini') or os.environ.get('GWS_TEST_INI')
    OPTIONS.update(load_options(ini_path))

    OPTIONS.update(dict(
        arg_ini=ini_path,
        arg_manifest=args.get('manifest'),
        arg_detach=args.get('d') or args.get('detach'),
        arg_coverage=args.get('c') or args.get('coverage'),
        arg_only=args.get('o') or args.get('only'),
        arg_verbose=args.get('v') or args.get('verbose'),
        arg_pytest=args.get('_rest'),
    ))

    OPTIONS['APP_DIR'] = APP_DIR

    p = OPTIONS.get('runner.base_dir')
    if not os.path.isabs(p):
        p = os.path.realpath(os.path.join(APP_DIR, p))
    OPTIONS['BASE_DIR'] = p

    if cmd == 'go':
        OPTIONS['arg_coverage'] = True
        OPTIONS['arg_detach'] = True
        compose_stop()
        configure()
        compose_start()
        run()
        compose_stop()
        return 0

    if cmd == 'start':
        compose_stop()
        configure()
        compose_start()
        return 0

    if cmd == 'stop':
        compose_stop()
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
    ensure_dir(f'{base}/postgres')
    ensure_dir(f'{base}/pytest_cache')

    ensure_dir(f'{base}/coverage', clear=True)

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

    run_args = []

    if OPTIONS['arg_only']:
        run_args.append(f'--only ' + OPTIONS['arg_only'])
    if OPTIONS['arg_verbose']:
        run_args.append(f'--verbose')
    if OPTIONS['arg_pytest']:
        run_args.append('-')
        run_args.extend(OPTIONS['arg_pytest'])

    docker_exec = f'''
        {OPTIONS.get('runner.docker')}
        exec
        {OPTIONS.get('runner.docker_exec_options', '')}
        c_gws
    '''

    prog = 'python3'
    cov_ini = f'{base}/config/coverage.ini'

    if OPTIONS['arg_coverage']:
        prog = f'coverage run --rcfile={cov_ini}'

    run_cmd = f'''
        {docker_exec} 
        {prog}
        /gws-app/gws/test/container_runner.py
        --base {OPTIONS['BASE_DIR']}
        {' '.join(run_args)}
    '''

    cli.run(run_cmd)

    if OPTIONS['arg_coverage']:
        ensure_dir(f'{base}/coverage', clear=True)
        html_cmd = f'''
            {docker_exec}  
            coverage html --rcfile={cov_ini}
        '''
        cli.run(html_cmd)


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
        srv.setdefault('container_name', f'c_{s}')

        vols = [
            f'{base}:{base}',
            f'{APP_DIR}:/gws-app',
            f'{base}/data:/data',
            f'{base}/gws-var:/gws-var',
        ]
        srv.setdefault('volumes', []).extend(vols)

        srv.setdefault('tmpfs', []).append('/tmp')
        srv.setdefault('stop_grace_period', '1s')

        srv.setdefault('environment', {})
        srv['environment'].update(OPTIONS.get('environment', {}))

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


def compose_start():
    base = OPTIONS['BASE_DIR']
    d = '--detach' if OPTIONS['arg_detach'] else ''
    cli.run(f'''{OPTIONS.get('runner.docker_compose')} --file {base}/config/docker-compose.yml up {d}''')


def compose_stop():
    base = OPTIONS['BASE_DIR']
    try:
        cli.run(f'''{OPTIONS.get('runner.docker_compose')} --file {base}/config/docker-compose.yml down''')
    except:
        pass


##

def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def write_file(path, s):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(s)


def load_options(local_ini):
    inis = [APP_DIR + '/test.ini']
    if local_ini:
        inis.append(local_ini)

    cli.info(f'using configs: {inis}')
    OPTIONS = inifile.from_paths(*inis)

    env = {
        'PYTHONPATH': '/gws-app',
    }
    for k, v in OPTIONS.items():
        sec, _, name = k.partition('.')
        if sec == 'environment':
            env[name] = v
    OPTIONS['environment'] = env

    return OPTIONS


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

def service_gws():
    tz = OPTIONS.get('service.gws.time_zone')
    if tz:
        command = f'bash -c "ln -fs /usr/share/zoneinfo/{tz} /etc/localtime && sleep infinity" '
    else:
        command = 'sleep infinity'

    return dict(
        command=command,
        ports=[
            f"{OPTIONS.get('service.gws.http_expose_port')}:80",
            f"{OPTIONS.get('service.gws.mpx_expose_port')}:5000",
        ],
    )


def service_qgis():
    return dict(
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


def service_postgres():
    # https://github.com/docker-library/docs/blob/master/postgres/README.md

    base = OPTIONS['BASE_DIR']
    write_file(f'{base}/postgres/postgresql.conf', _POSTGRESQL_CONF)

    ensure_dir(f'{base}/postgres')

    return dict(
        environment={
            'POSTGRES_DB': OPTIONS.get('service.postgres.database'),
            'POSTGRES_PASSWORD': OPTIONS.get('service.postgres.password'),
            'POSTGRES_USER': OPTIONS.get('service.postgres.user'),
        },
        ports=[
            f"{OPTIONS.get('service.postgres.expose_port')}:5432",
        ],
        volumes=[
            f"{base}/postgres:/var/lib/postgresql/data"
        ]
    )


def service_mockserver():
    return dict(
        # NB use the gws image
        image=OPTIONS.get('service.gws.image'),
        command=f'python3 /gws-app/gws/test/mockserver.py',
        ports=[
            f"{OPTIONS.get('service.mockserver.expose_port')}:80",
        ],
    )


if __name__ == '__main__':
    cli.main('test', main, USAGE)
