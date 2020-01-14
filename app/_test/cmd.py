"""Test runner support."""

"""
Each test suite (_test/suite/<suite-name>) runs in its own dedicated container. It has its own file system 
with "/data" "/data/config.cx" etc

Containers are managed by a simple server which responds to http requests like 'server?suite=<suite-name>'.

The server also starts separate postgres and qgis containers.

To run the tests:
    - start the server (cmd.py server), runs in the foreground
    - (in a new terminal) run a suite (cmd.py run) or all of them (cmd.py batch)
"""

import argparse
import atexit
import http.server
import json
import os
import re
import requests
import subprocess
import sys
import time
import urllib.parse

CONFIG = {}

TEST_DIR = "_test"
SUITE_DIR = "suite"

color = {
    'black': '\u001b[30m',
    'red': '\u001b[31m',
    'green': '\u001b[32m',
    'yellow': '\u001b[33m',
    'blue': '\u001b[34m',
    'magenta': '\u001b[35m',
    'cyan': '\u001b[36m',
    'white': '\u001b[37m',
    'reset': '\u001b[0m',
}


def banner(s):
    sys.stderr.write(color['magenta'] + '>>>>>>> ' + s + color['reset'] + '\n')
    sys.stderr.flush()


def run(cmd, **kwargs):
    args = {
        'stderr': subprocess.STDOUT,
        'shell': True,
    }
    args.update(kwargs)

    wait = args.pop('wait', True)

    banner(cmd)
    p = subprocess.Popen(cmd, **args)
    if not wait:
        return 0
    p.communicate()
    return p.returncode


def docker_run(image, opts, cmd=''):
    run(f"docker run {' '.join(opts)} {image} {cmd}", wait=False)


def start_suite_container(suite, extra_options=None):
    stop_suite_container()

    opts = [
        f"--add-host={CONFIG['host.name']}:{CONFIG['host.docker_ip']}",
        f"--env GWS_CONFIG=/data/config.cx",
        f"--env GWS_TMP_DIR=/gws-var/tmp",
        f"--mount type=bind,src={CONFIG['paths.app_root']},dst=/gws-app",
        f"--mount type=bind,src={CONFIG['paths.app_root']}/_test/common,dst=/common",
        f"--mount type=bind,src={CONFIG['paths.app_root']}/_test/suite/{suite}/data,dst=/data",
        f"--mount type=bind,src={CONFIG['paths.suite_var_root']},dst=/gws-var",
        f"--name {CONFIG['suite_container.name']}",
        f"--publish {CONFIG['host.ip']}:{CONFIG['suite_container.http_port']}:80",
        # publish the bundled qgis for debugging
        f"--publish {CONFIG['host.ip']}:{CONFIG['suite_container.qgis_port']}:4000",
    ]

    opts += CONFIG['suite_container.options']

    if extra_options:
        opts += extra_options

    # write a copy of our config so that the container can read it

    with open(CONFIG['paths.suite_var_root'] + '/test.config.json', 'w') as fp:
        json.dump(CONFIG, fp)

    # startup script for the container

    start_script = f"""
        if [ -f /gws-app/_test/suite/{suite}/init.py ]; then
            PYTHONPATH=/gws-app python /gws-app/_test/suite/{suite}/init.py
        fi
        if [ $? -ne 0 ]; then
            exit
        fi
        /gws-app/bin/gws server start -v
    """

    with open(CONFIG['paths.suite_var_root'] + '/start.sh', 'w') as fp:
        fp.write(start_script.strip())

    # ready to go

    banner('CONTAINER START')
    docker_run(CONFIG['suite_container.image'], opts, 'bash /gws-var/start.sh')


def stop_suite_container():
    banner('CONTAINER STOP')
    run(f"docker kill --signal SIGINT {CONFIG['suite_container.name']}")
    run(f"docker rm --force {CONFIG['suite_container.name']}")
    run(f"rm -fr {CONFIG['paths.suite_var_root']}/*")


def start_postgres():
    # see https://hub.docker.com/r/kartoza/postgis/ for details

    stop_postgres()
    time.sleep(2)

    extra = '\\n'.join([
        "log_destination='stderr'",
        "log_statement='all'",
        "log_duration=1",
    ])

    opts = [
        f"--detach",
        f"--env POSTGRES_DB={CONFIG['postgres_connection.database']}",
        f"--env POSTGRES_PASS={CONFIG['postgres_connection.password']}",
        f"--env POSTGRES_USER={CONFIG['postgres_connection.user']}",
        f"""--env EXTRA_CONF="{extra}" """
        f"--name {CONFIG['postgres_container.name']}",
        f"--publish {CONFIG['host.ip']}:{CONFIG['postgres_connection.port']}:5432",
    ]

    banner('STARTING POSTGRES...')
    docker_run(CONFIG['postgres_container.image'], opts)


def stop_postgres():
    banner('STOPPING POSTGRES...')
    run(f"docker kill {CONFIG['postgres_container.name']}")
    run(f"docker rm --force {CONFIG['postgres_container.name']}")


def start_qgis():
    # we use our own image to expose the qgis server which will provide us with test OWS services
    # port 4000 (our qgis port) is exposed to the host
    # the server serves projects from _test/common/qgis
    # qgs project files must be patched to provide correct callback urls (host:qgis-port)

    stop_qgis()
    time.sleep(2)

    opts = [
        f"--add-host={CONFIG['host.name']}:{CONFIG['host.docker_ip']}",
        f"--detach",
        f"--env GWS_CONFIG=/qgis/config.cx",
        f"--env GWS_TMP_DIR=/gws-var/tmp",
        f"--mount type=bind,src={CONFIG['paths.app_root']},dst=/gws-app",
        f"--mount type=bind,src={CONFIG['paths.app_root']}/_test/common/qgis,dst=/qgis",
        f"--mount type=bind,src={CONFIG['paths.qgis_var_root']},dst=/gws-var",
        f"--name {CONFIG['qgis_container.name']}",
        f"--publish {CONFIG['host.ip']}:{CONFIG['qgis_container.port']}:4000",
    ]

    banner('STARTING QGIS...')
    docker_run(CONFIG['qgis_container.image'], opts, 'bash /gws-app/bin/gws server start -v')


def stop_qgis():
    banner('STOPPING QGIS...')
    run(f"docker kill {CONFIG['qgis_container.name']}")
    run(f"docker rm --force {CONFIG['qgis_container.name']}")
    run(f"rm -fr {CONFIG['paths.qgis_var_root']}/*")


def stop_all():
    stop_suite_container()
    stop_postgres()
    stop_qgis()


##

def start_container_for_suite(suite, extra_options=None):
    run(f"make -C {CONFIG['paths.app_root']}/.. spec")
    start_suite_container(suite, extra_options)


def exec_suite(suite, opts):
    run(f"docker exec {CONFIG['suite_container.name']} pytest /gws-app/_test/suite/{suite} {opts or ''}")


def run_suite(suite, opts):
    banner(f'RUNNING {suite}...')

    banner('STARTING CONTAINER...')

    start_url = f"http://{CONFIG['host.ip']}:{CONFIG['cmdserver.port']}?suite={suite}"
    requests.get(start_url)

    wait_cnt = 0

    while True:
        try:
            requests.get(f"http://{CONFIG['host.ip']}:{CONFIG['suite_container.http_port']}")
            break
        except Exception as e:
            wait_cnt += 1
            if wait_cnt < 10:
                banner(f"WAITING FOR THE CONTAINER: {e.__class__.__name__}")
                time.sleep(5)
            else:
                banner('GIVING UP')
                return

    exec_suite(suite, opts)


def run_suites(suites, opts):
    if suites:
        suites = [s.strip() for s in suites.split(',')]
    else:
        suites = []
        base = f"{CONFIG['paths.app_root']}/_test/suite"
        for p in os.listdir(base):
            if os.path.isdir(base + '/' + p):
                suites.append(p)

    for suite in sorted(suites):
        run_suite(suite, opts)


class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        params = {}
        if '?' in self.path:
            params = dict(urllib.parse.parse_qsl(self.path.split('?')[1]))

        if 'suite' in params:
            start_container_for_suite(params['suite'], ['-i'])

        if 'exit' in params:
            banner('STOPPING CMD SERVER')
            sys.exit(0)

        self.send_response(200)
        self.send_header("Content-type", 'text/plain')
        self.send_header("Content-Length", 0)
        self.end_headers()


def start_cmd_server():
    try:
        # signal the server to exit
        stop_url = f"http://{CONFIG['host.ip']}:{CONFIG['cmdserver.port']}?exit=1"
        requests.get(stop_url)
    except:
        pass

    atexit.register(stop_all)

    start_postgres()
    start_qgis()

    server_address = CONFIG['host.ip'], int(CONFIG['cmdserver.port'])

    HTTPRequestHandler.protocol_version = 'HTTP/1.1'
    httpd = http.server.HTTPServer(server_address, HTTPRequestHandler)

    sa = httpd.socket.getsockname()
    banner(f'STARTED CMD SERVER ON {sa[0]}:{sa[1]}')
    httpd.serve_forever()


def parse_config(path):
    d = {}
    with open(path) as fp:
        for s in fp:
            s = s.strip()
            if not s or s.startswith('#'):
                continue
            a, b = s.split('=')
            d[a.strip()] = b.strip()
    return d


def main():
    global CONFIG

    usage = """
    
        cmd.py server --config CONFIG-PATH
    
            start a simple HTTP server which run gws containers on demand: curl localhost:port?suite=SUITE
    
        cmd.py run --config CONFIG-PATH --suite SUITE-NAMES --opts 'PYTEST-OPTIONS'
    
            run each suite from a comma separated list (or all if omitted)
    
        cmd.py exec --config CONFIG-PATH --suite SUITE-NAME --opts 'PYTEST-OPTIONS'
    
            run pytest on a suite, assuming its container is started
    
    """

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('cmd', help='command')
    parser.add_argument('--config', dest='config', help='configuration file path')
    parser.add_argument('--suite', dest='suite', help='suite name')
    parser.add_argument('--opts', dest='opts', help='pytest options')

    args = parser.parse_args()
    args.opts = (args.opts or '').strip("'")
    args.suite = (args.suite or '').split('/')[-1]

    CONFIG = parse_config(os.path.dirname(__file__) + '/cmd.ini')
    if args.config:
        CONFIG.update(parse_config(args.config))

    if args.cmd == 'server':
        start_cmd_server()
        return

    if args.cmd in ('run', 'r'):
        run_suites(args.suite, args.opts)
        return

    if args.cmd in ('exec', 'e'):
        exec_suite(args.suite, args.opts)
        return


if __name__ == '__main__':
    main()
