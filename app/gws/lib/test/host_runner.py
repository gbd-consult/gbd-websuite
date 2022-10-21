"""Test configurator and invoker.

This file provides utilities for the test runner
on the host machine (see `make test` and `app/test.py`).
The purpose of these utils is to create a docker compose file, start the compose
and invoke the test runner inside the GWS container (`gws test`).
"""

import configparser
import json
import os
import shutil
import subprocess
import sys

import yaml

HELP = """
GWS test runner
~~~~~~~~~~~~~~~

    python3 test.py <command> [<pattern>] [--manifest <manifest>] [pytest options]

Commands:

    configure  - configure the test environment
    go         - start the test environment, run tests and stop
    run        - run tests
    restart    - restart the test environment
    start      - start the test environment
    stop       - stop the test environment
        
Options:

    <pattern>  - only run test files whose name contains the pattern
    <manifest> - path to MANIFEST.json
    
    Additionaly, you can pass any pytest option:
    https://docs.pytest.org/en/6.2.x/reference.html#command-line-flags

"""

APP_DIR = os.path.realpath(os.path.dirname(__file__) + '/../../../')

TEST_CONFIG_PATH_IN_CONTAINER = '/gws-var/TEST_CONFIG.json'

CONFIG = {}


def main():
    args = sys.argv[1:]

    if '-h' in args or '--help' in args:
        print(HELP)
        return 0

    CONFIG.update(parse_ini(
        APP_DIR + '/test.ini',
        APP_DIR + '/___local.test.ini'
    ))

    CONFIG.setdefault('compose_yaml_path', CONFIG['runner.work_dir'] + '/docker-compose.yml')

    CONFIG.setdefault('manifest', None)
    if '--manifest' in args:
        n = args.index('--manifest')
        p = args[n + 1]
        if p:
            CONFIG['manifest'] = read_file(p)
        args = args[:n] + args[n + 2:]

    cmd = None
    if args:
        cmd = args.pop(0)

    if cmd == 'configure':
        configure_all()
        sys.exit(0)

    if cmd == 'go':
        compose_stop()
        configure_all()
        compose_start()
        runner_run(args)
        sys.exit(0)

    if cmd == 'run':
        runner_configure()
        runner_run(args)
        sys.exit(0)

    if cmd == 'restart':
        compose_stop()
        configure_all()
        compose_start()
        sys.exit(0)

    if cmd == 'start':
        configure_all()
        compose_start()
        sys.exit(0)

    print(HELP)
    sys.exit(255)


#

def configure_all():
    reset_work_dir()
    compose_configure()
    runner_configure()


def reset_work_dir():
    wd = CONFIG['runner.work_dir']
    de: os.DirEntry
    for de in os.scandir(wd):
        if de.is_dir():
            shutil.rmtree(de.path)
        else:
            os.unlink(de.path)


def runner_configure():
    wd = CONFIG['runner.work_dir']

    ensure_dir(f'{wd}/gws-var')
    ensure_dir(f'{wd}/gws-tmp')
    ensure_dir(f'{wd}/web')

    CONFIG['pytest_ini_path'] = '/gws-var/pytest.ini'
    create_pytest_ini(wd + CONFIG['pytest_ini_path'])

    write_file(f"{wd}/{TEST_CONFIG_PATH_IN_CONTAINER}", json.dumps(CONFIG, indent=4))


def runner_run(args):
    run_cmd(f'''docker exec {CONFIG['service.gws.container_name']} gws test ''' + ' '.join(args))


def compose_configure():
    write_file(
        CONFIG['compose_yaml_path'],
        yaml.dump(compose_config()))


def compose_start():
    run_cmd(f'''docker-compose --file {CONFIG['compose_yaml_path']} up --detach''')


def compose_stop():
    try:
        run_cmd(f'''docker-compose --file {CONFIG['compose_yaml_path']} down''')
    except:
        pass


def compose_config():
    wd = CONFIG['runner.work_dir']

    services = {}

    for s in CONFIG['runner.services'].split():
        cfg = globals()[f'compose_config_for_service_{s}']()
        cname = CONFIG[f'service.{s}.container_name']

        cfg.setdefault('image', CONFIG[f'service.{s}.image'])

        cfg.setdefault('extra_hosts', [
            f"{CONFIG['runner.host_name']}:{CONFIG['runner.container_host_ip']}",
            f"qgis:{CONFIG['runner.container_host_ip']}"
        ])

        cfg.setdefault('container_name', cname)
        cfg.setdefault('volumes', []).append(f"{wd}:{wd}")

        services[cname] = cfg

    return {
        'version': '3',
        'services': services,
    }


# services

def compose_config_for_service_gws():
    wd = CONFIG['runner.work_dir']

    ensure_dir(f'{wd}/gws-var')

    bootstrap_cfg = {
        'server': {
            'mapproxy': {'enabled': True, 'workers': 1, 'forceStart': True},
            'monitor': {'enabled': False},
            'log': {'level': 'DEBUG'},
            'qgis': {'enabled': True, 'workers': 1},
            'spool': {'enabled': True, 'workers': 1},
            'web': {'enabled': True, 'workers': 1},
            'timeout': 60,
        },
    }

    bootstrap_cfg_path = '/gws-var/bootstrap_config.json'

    write_file(f'{wd}/{bootstrap_cfg_path}', json.dumps(bootstrap_cfg, indent=4))

    return {
        'ports': [
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.gws.http_port']}:80",
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.gws.mpx_port']}:5000",
            f"{CONFIG['runner.host_ip']}:2222:2222",
        ],
        'command': 'sleep infinity',
        'environment': {
            'GWS_CONFIG': bootstrap_cfg_path,
        },
        'volumes': [
            f"{APP_DIR}:{APP_DIR}",
            f"{APP_DIR}:/gws-app",
            f"{wd}/gws-var:/gws-var",
            f"{wd}/gws-tmp:/tmp",
        ],
    }


def compose_config_for_service_postgres():
    # https://hub.docker.com/r/kartoza/postgis

    extra_conf = r"log_destination='stderr'\nlog_statement='all'\nlog_duration=1"

    return {
        'environment': {
            'POSTGRES_DB': CONFIG['service.postgres.database'],
            'POSTGRES_PASS': CONFIG['service.postgres.password'],
            'POSTGRES_USER': CONFIG['service.postgres.user'],
            'EXTRA_CONF': extra_conf,
        },
        'ports': [
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.postgres.port']}:5432",
        ],
    }


def compose_config_for_service_mockserv():
    wd = CONFIG['runner.work_dir']
    server_app = 'mockserv.app.py'
    reqs_file = 'mockserv_requirements.txt'

    write_file(
        f"{wd}/{server_app}",
        read_file(f'{APP_DIR}/gws/lib/test/{server_app}')
    )

    write_file(
        f"{wd}/{reqs_file}",
        '\n'.join(CONFIG['service.mockserv.requirements'].split()) + '\n'
    )

    command = f"bash -c 'pip install -r {wd}/{reqs_file} && python {wd}/{server_app}'"

    return {
        'command': command,
        'ports': [
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.mockserv.port']}:8080",
        ],
    }


def compose_config_for_service_qgis():
    command = f"/bin/sh /qgis-start.sh"

    return {
        'command': command,
        'ports': [
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.qgis.port']}:80",
        ],
    }


# utils

def parse_ini(*paths):
    cfg = {}
    cc = configparser.ConfigParser()

    for path in paths:
        if os.path.isfile(path):
            cc.read(path)

    for sec in cc.sections():
        for opt in cc.options(sec):
            cfg[sec + '.' + opt] = cc.get(sec, opt)

    return cfg


def create_pytest_ini(path):
    wd = CONFIG['runner.work_dir']

    cc = configparser.ConfigParser()
    cc.add_section('pytest')

    for k, v in CONFIG.items():
        k = k.split('.')
        if k[0] == 'pytest':
            cc.set('pytest', k[1], v)

    cc.set('pytest', 'python_files', '*_test.py')
    cc.set('pytest', 'cache_dir', f'{wd}/pytest_cache')

    with open(path, 'wt') as fp:
        cc.write(fp)


def run_cmd(cmd, **kwargs):
    args = {
        'stderr': subprocess.STDOUT,
        'shell': True,
    }
    args.update(kwargs)

    wait = args.pop('wait', True)

    print('>', cmd)
    p = subprocess.Popen(cmd, **args)
    if not wait:
        return 0
    p.communicate()
    return p.returncode


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_file(path, text):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()
