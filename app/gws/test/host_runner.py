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

    python3 test/host_runner.py <command> <options> - <pytest options>

Commands:

    go         - configure, start the test environment, run tests and stop
    configure  - configure the test environment
    start      - configure and start the test environment
    stop       - stop the test environment
    restart    - restart the test environment
    run        - run tests in a started environment

options:

    --ini <path>          - path to the local 'ini' file (can also be passed in the GWS_TEST_INI env var)
    --manifest <manifest> - path to MANIFEST.json
    --only <regex>        - only run filenames matching the pattern 
    --verbose             - enable debug logging
    
Pytest options:
    see https://docs.pytest.org/latest/reference.html#command-line-flags

"""


def main(args):
    cmd = args.get(1)

    options = load_options(args.get('ini') or os.environ.get('GWS_TEST_INI'))
    options['APP_DIR'] = APP_DIR

    if cmd == 'configure':
        do_configure(options, args)
        return 0

    if cmd == 'go':
        compose_stop(options)
        do_configure(options, args)
        compose_start(options, detach=True)
        do_run(options, args)
        compose_stop(options)
        return 0

    if cmd == 'start':
        compose_stop(options)
        do_configure(options, args)
        compose_start(options)
        return 0

    if cmd == 'restart':
        compose_stop(options)
        compose_start(options)
        return 0

    if cmd == 'stop':
        compose_stop(options)
        return 0

    if cmd == 'run':
        do_run(options, args)
        return 0

    cli.fatal('invalid arguments, try test.py -h for help')


##

def do_configure(options, args):
    wd = options.get('runner.work_dir')
    make_dir(wd)
    clear_dir(wd)

    manifest_text = '{}'
    s = args.get('manifest')
    if s:
        manifest_text = read_file(s)

    write_file(f'{wd}/MANIFEST.json', manifest_text)
    write_file(f'{wd}/docker-compose.yml', make_docker_compose_yml(options))
    write_file(f'{wd}/pg_service.conf', make_pg_service_conf(options))
    write_file(f'{wd}/pytest.ini', make_pytest_ini(options))
    write_file(f'{wd}/options.json', json.dumps(options, indent=4))

    cli.info(f'tests configured in {wd!r}')


def do_run(options, args):
    wd = options.get('runner.work_dir')
    cmd = ' '.join([
        options.get('runner.docker'),
        'exec',
        options.get('runner.exec_options', ''),
        options.get('service.gws.name'),
        'gws test',
        '--work_dir',
        wd,
        *sys.argv[2:],
    ])
    cli.run(cmd)


##


def make_docker_compose_yml(options):
    wd = options.get('runner.work_dir')

    service_configs = {}

    services = options.get('runner.services').split(',')
    if 'all' in services:
        services = [k.split('_')[1] for k in globals() if k.startswith('service_')]
    options['runner.services'] = services

    for s in services:
        srv = globals().get(f'service_{s}')(options)

        srv.setdefault('image', options.get(f'service.{s}.image'))
        srv.setdefault('extra_hosts', []).append(f"{options.get('runner.docker_host_name')}:host-gateway")
        srv.setdefault('container_name', options.get(f'service.{s}.name'))
        srv.setdefault('volumes', []).append(f"{wd}:{wd}")
        srv.setdefault('tmpfs', []).append('/tmp')
        srv.setdefault('stop_grace_period', '1s')
        srv.setdefault('environment', {})

        for k, v in options.get('environment').items():
            srv['environment'].setdefault(k, v)

        service_configs[s] = srv

    cfg = {
        'version': '3',
        'networks': {
            'default': {
                'name': 'gws_test_network'
            }
        },
        'services': service_configs,
    }

    return yaml.dump(cfg)


def make_pg_service_conf(options):
    name = options.get('service.postgres.name')
    ini = {
        f'{name}.host': options.get('service.postgres.host'),
        f'{name}.port': options.get('service.postgres.port'),
        f'{name}.user': options.get('service.postgres.user'),
        f'{name}.password': options.get('service.postgres.password'),
        f'{name}.dbname': options.get('service.postgres.database'),
    }
    return inifile.to_string(ini)


def make_pytest_ini(options):
    wd = options.get('runner.work_dir')

    ini = {
        'pytest.cache_dir': f'{wd}/pytest_cache'
    }
    for k, v in options.items():
        if k.startswith('pytest.'):
            ini[k] = v
    return inifile.to_string(ini)


def compose_start(options, detach=False):
    wd = options.get('runner.work_dir')
    d = '--detach' if detach else ''
    cli.run(f'''{options.get('runner.docker_compose')} --file {wd}/docker-compose.yml up {d}''')


def compose_stop(options):
    wd = options.get('runner.work_dir')
    try:
        cli.run(f'''{options.get('runner.docker_compose')} --file {wd}/docker-compose.yml down''')
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
    options = inifile.from_paths(*inis)

    env = {}
    for k, v in options.items():
        sec, _, name = k.partition('.')
        if sec == 'environment':
            if v.startswith('./'):
                v = options.get('runner.work_dir') + v[1:]
            env[name] = v
    options['environment'] = env

    for k, v in env.items():
        os.environ[k] = v

    return options


def make_dir(d):
    os.makedirs(d, exist_ok=True)


def clear_dir(d):
    for de in os.scandir(d):
        if de.is_dir():
            clear_dir(de.path)
        else:
            os.unlink(de.path)


##

def service_gws(options):
    wd = options.get('runner.work_dir')

    make_dir(f'{wd}/gws-var')
    make_dir(f'{wd}/gws-tmp')
    make_dir(options.get('service.gws.data_dir'))

    return {
        'command': 'sleep infinity',
        'ports': [
            f"{options.get('service.gws.http_expose_port')}:80",
            f"{options.get('service.gws.mpx_expose_port')}:5000",
        ],
        'volumes': [
            f"{APP_DIR}:/gws-app",
            f"{wd}/gws-var:/gws-var",
            # f"{wd}/gws-tmp:/tmp",
            f"{options.get('service.gws.data_dir')}:/data",
        ],
    }


def service_qgis(options):
    return {
        'command': f'/bin/sh /qgis-start.sh',
        'ports': [
            f"{options.get('service.qgis.expose_port')}:80",
        ],
    }


def service_postgres(options):
    # https://github.com/docker-library/docs/blob/master/postgres/README.md

    make_dir(options.get('service.postgres.data_dir'))

    return {
        'environment': {
            'POSTGRES_DB': options.get('service.postgres.database'),
            'POSTGRES_PASSWORD': options.get('service.postgres.password'),
            'POSTGRES_USER': options.get('service.postgres.user'),
        },
        'ports': [
            f"{options.get('service.postgres.expose_port')}:5432",
        ],
        'volumes': [
            f"{options.get('service.postgres.data_dir')}:/var/lib/postgresql/data",
        ],
    }


def service_mockserver(options):
    wd = options.get('runner.work_dir')

    server_app = 'mockserver.py'
    write_file(f'{wd}/{server_app}', read_file(f'{APP_DIR}/gws/test/{server_app}'))

    return {
        # NB use the gws image
        'image': options.get('service.gws.image'),
        'ports': [
            f"{options.get('service.mockserver.expose_port')}:80",
        ],
        'command': f'python3 {wd}/{server_app}',
    }


if __name__ == '__main__':
    cli.main('test', main, USAGE)
