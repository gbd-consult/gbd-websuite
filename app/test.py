import configparser
import json
import os
import subprocess
import sys

import yaml

APP_DIR = os.path.realpath(os.path.dirname(__file__))

CONFIG = {}

HELP = """
GWS test runner.

Usage (on the host machine):

    python3 test.py go [--manifest <manifest-path>] <test options>
        start the docker compose environment, run tests and stop

    python3 test.py start [--manifest <manifest-path>]
        prepare and start the docker compose environment

    python3 test.py stop [--manifest <manifest-path>]
        stop the docker compose environment
        
    python3 test.py configure [--manifest <manifest-path>]
        configure tests

    python3 test.py run [--manifest <manifest-path>] <test options>
        run tests
        
Test options are: [pattern] [pytest-options]    

    - pattern        : only run test files that contain the pattern
    - pytest-options : https://docs.pytest.org/en/6.2.x/reference.html#command-line-flags
"""


def main():
    args = sys.argv[1:]

    fn = None
    if args:
        fn = globals().get('cmd_' + args.pop(0))

    if not fn:
        print(HELP)
        return 1

    CONFIG['MANIFEST'] = None

    CONFIG.update(parse_ini(
        APP_DIR + '/test.ini',
        APP_DIR + '/___local.test.ini'
    ))

    if args and args[0] == '--manifest':
        # @TODO must use spec/manifest
        args.pop(0)
        if args and args[0]:
            CONFIG['MANIFEST'] = json.loads(read_file(args.pop(0)))

    return fn(args)


def cmd_go(args):
    compose_stop()
    compose_configure()
    compose_start()
    runner_configure()
    runner_run(args)
    compose_stop()


def cmd_start(args):
    compose_configure()
    compose_start()


def cmd_stop(args):
    compose_stop()


def cmd_restart(args):
    compose_stop()
    compose_configure()
    compose_start()


def cmd_run(args):
    runner_configure()
    runner_run(args)


def cmd_configure(args):
    compose_configure()
    runner_configure()


#

def runner_configure():
    wd = CONFIG['runner.work_dir']

    ensure_dir(f'{wd}/gws-var')
    ensure_dir(f'{wd}/gws-tmp')
    ensure_dir(f'{wd}/web')

    CONFIG['PATH_TO_PYTEST_INI'] = '/gws-var/PYTEST.ini'
    create_pytest_ini(wd + CONFIG['PATH_TO_PYTEST_INI'])

    write_file(f"{wd}/gws-var/TEST_CONFIG.json", json.dumps(CONFIG, indent=4))


def runner_run(args):
    run_cmd(f'''docker exec {CONFIG['service.gws.container_name']} gws test ''' + ' '.join(args))


#

def compose_configure():
    write_file(
        compose_yaml_path(),
        yaml.dump(compose_config()))


def compose_start():
    path = compose_yaml_path()
    run_cmd(f'''docker-compose --file {path} up --detach''')


def compose_stop():
    path = compose_yaml_path()
    try:
        run_cmd(f'''docker-compose --file {path} down''')
    except:
        pass


def compose_yaml_path():
    return CONFIG['runner.work_dir'] + '/DOCKER_COMPOSE.yml'


def compose_config():
    services = {}

    for s in CONFIG['runner.services'].split():
        cfg = globals()[f'{s}_service_config']()
        cname = CONFIG[f'service.{s}.container_name']

        cfg.setdefault('image', CONFIG[f'service.{s}.image'])

        cfg.setdefault('extra_hosts', [
            f"{CONFIG['runner.host_name']}:{CONFIG['runner.container_host_ip']}",
            f"qgis:{CONFIG['runner.container_host_ip']}"
        ])

        cfg.setdefault('container_name', cname)
        cfg.setdefault('volumes', []).append(
            f"{CONFIG['runner.work_dir']}:{CONFIG['runner.work_dir']}"
        )

        services[cname] = cfg

    return {
        'version': '3',
        'services': services,
    }


# services

def gws_service_config():
    wd = CONFIG['runner.work_dir']

    ensure_dir(f'{wd}/gws-var')

    gws_config = {
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

    write_file(f'{wd}/gws-var/INITIAL_GWS_CONFIG.json', json.dumps(gws_config, indent=4))

    return {
        'ports': [
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.gws.http_port']}:80",
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.gws.qgis_port']}:4000",
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.gws.mpx_port']}:5000",
            f"{CONFIG['runner.host_ip']}:2222:2222",
        ],
        'command': CONFIG['service.gws.command'],
        'environment': {
            'GWS_CONFIG': '/gws-var/INITIAL_GWS_CONFIG.json',
        },
        'volumes': [
            f"{APP_DIR}:{APP_DIR}",
            f"{APP_DIR}:/gws-app",
            f"{wd}/gws-var:/gws-var",
            f"{wd}/gws-tmp:/tmp",
        ],
    }


def postgres_service_config():
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


def web_service_config():
    wd = CONFIG['runner.work_dir']

    code = read_file(APP_DIR + '/gws/lib/test_web_server.py')
    write_file(f"{wd}/test_web_server.py", code)

    reqs = '\n'.join(CONFIG['service.web.requirements'].split())
    write_file(f"{wd}/test_web_requirements.txt", reqs + '\n')

    command = f"bash -c 'pip install -r {wd}/test_web_requirements.txt && python {wd}/test_web_server.py'"

    return {
        'command': command,
        'ports': [
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.web.port']}:8080",
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


if __name__ == '__main__':
    sys.exit(main() or 0)
