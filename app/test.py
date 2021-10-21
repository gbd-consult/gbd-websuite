import configparser
import json
import os
import subprocess
import shutil
import sys

import yaml

APP_DIR = os.path.realpath(os.path.dirname(__file__))

CONFIG = {}

HELP = """

GWS test runner
~~~~~~~~~~~~~~~

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

    CONFIG['COMPOSE_YAML_PATH'] = CONFIG['runner.work_dir'] + '/docker-compose.yml'

    return fn(args)


def cmd_stop(args):
    compose_stop()


def cmd_configure(args):
    reset_work_dir()
    compose_configure()
    runner_configure()


def cmd_go(args):
    cmd_restart(args)
    runner_run(args)
    cmd_stop(args)


def cmd_start(args):
    cmd_configure(args)
    compose_start()


def cmd_restart(args):
    cmd_stop(args)
    cmd_start(args)


def cmd_run(args):
    runner_configure()
    runner_run(args)


#

def runner_configure():
    wd = CONFIG['runner.work_dir']

    ensure_dir(f'{wd}/gws-var')
    ensure_dir(f'{wd}/gws-tmp')
    ensure_dir(f'{wd}/web')

    CONFIG['PYTEST_INI_PATH'] = '/gws-var/PYTEST.ini'
    create_pytest_ini(wd + CONFIG['PYTEST_INI_PATH'])

    write_file(f"{wd}/gws-var/TEST_CONFIG.json", json.dumps(CONFIG, indent=4))


def runner_run(args):
    run_cmd(f'''docker exec {CONFIG['service.gws.container_name']} gws test ''' + ' '.join(args))


#

def compose_configure():
    write_file(
        CONFIG['COMPOSE_YAML_PATH'],
        yaml.dump(compose_config()))


def compose_start():
    run_cmd(f'''docker-compose --file {CONFIG['COMPOSE_YAML_PATH']} up --detach''')


def compose_stop():
    try:
        run_cmd(f'''docker-compose --file {CONFIG['COMPOSE_YAML_PATH']} down''')
    except:
        pass


def compose_config():
    wd = CONFIG['runner.work_dir']

    services = {}

    for s in CONFIG['runner.services'].split():
        cfg = globals()[f'service_{s}_config']()
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

def service_gws_config():
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

    bootstrap_cfg_path = '/gws-var/BOOTSTRAP_CONFIG.json'

    write_file(f'{wd}/{bootstrap_cfg_path}', json.dumps(bootstrap_cfg, indent=4))

    return {
        'ports': [
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.gws.http_port']}:80",
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.gws.qgis_port']}:4000",
            f"{CONFIG['runner.host_ip']}:{CONFIG['service.gws.mpx_port']}:5000",
            f"{CONFIG['runner.host_ip']}:2222:2222",
        ],
        'command': CONFIG['service.gws.command'],
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


def service_postgres_config():
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


def service_web_config():
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


def reset_work_dir():
    wd = CONFIG['runner.work_dir']
    de: os.DirEntry
    for de in os.scandir(wd):
        if de.is_dir():
            shutil.rmtree(de.path)
        else:
            os.unlink(de.path)


def write_file(path, text):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


if __name__ == '__main__':
    sys.exit(main() or 0)
