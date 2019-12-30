"""Test runner support."""

"""
Each test suite (_test/suite/<suite-name>) runs in its own dedicated container. It has its own file system 
with "/data" "/data/config.cx" etc

Containers are managed by a simple server which responds to http requests like 'server?suite=<suite-name>'.

To run the tests:
    - start the postgres container (cmd.py postgres), runs in the background 
    - start the server (cmd.py server), runs in the foreground
    - run a suite (cmd.py run) or all of them (cmd.py batch)
"""

import argparse
import atexit
import http.server
import json
import os
import requests
import subprocess
import sys
import time
import urllib.parse

CONFIG = {}

# path to the suites relatively to the app root
SUITE_BASE = "_test/suite"

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


def cfg(key):
    a, b = key.split('.')
    return CONFIG[a][b]


def banner(s):
    print(color['magenta'] + '>>>>>>> ' + s + color['reset'])


def run(cmd, **kwargs):
    args = {
        'stdin': None,
        'stdout': None,
        'stderr': None,
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


def suite_root_dir(suite):
    return f"{cfg('paths.app_root')}/{SUITE_BASE}/{suite}"


def start_container(suite, extra_options=None):
    stop_container()

    opts = [
        f"--rm",
        f"--env GWS_CONFIG=/data/config.cx",
        f"--env GWS_TMP_DIR=/gws-var/tmp",
        f"--name {cfg('docker.container_name')}",
        f"--mount type=bind,src={cfg('paths.var_root')}/{suite}/data,dst=/data",
        f"--mount type=bind,src={cfg('paths.app_root')},dst=/gws-app",
        f"--mount type=bind,src={cfg('paths.var_root')},dst=/gws-var",
        f"--publish 0.0.0.0:{cfg('docker.http_port')}:80",
    ]

    opts += cfg('docker.options')

    if extra_options:
        opts += extra_options

    # write a copy of our config so that the container can read it

    with open(cfg('paths.var_root') + '/test.config.json', 'w') as fp:
        json.dump(CONFIG, fp)

    # create a copy of the suite dir so that the container can modify it (in init.py)

    suite_dir = suite_root_dir(suite)
    run(f"cp -r {suite_dir} {cfg('paths.var_root')}")
    run(f"touch {cfg('paths.var_root')}/{suite}/init.py")

    # startup script for the container

    start_script = f"PYTHONPATH=/gws-app python /gws-var/{suite}/init.py && /gws-app/bin/gws server start -v"

    with open(cfg('paths.var_root') + '/start.sh', 'w') as fp:
        fp.write(start_script.strip())

    # ready to go

    banner('CONTAINER START')
    run(f"docker run {' '.join(opts)} {cfg('docker.image_name')} bash /gws-var/start.sh", wait=False)


def stop_container():
    banner('CONTAINER STOP')
    run(f"docker kill --signal SIGINT {cfg('docker.container_name')}")
    run(f"docker rm --force {cfg('docker.container_name')}")
    run(f"rm -fr {cfg('paths.var_root')}/*")


def start_container_for_suite(suite, extra_options=None):
    run(f"make -C {cfg('paths.app_root')}/.. spec")
    start_container(suite, extra_options)


def exec_suite(suite, opts):
    run(f"docker exec {cfg('docker.container_name')} pytest /gws-app/{SUITE_BASE}/{suite} {opts or ''}")


def run_suite(suite, opts):
    banner(f'RUNNING {suite}...')

    banner('STARTING CONTAINER...')

    start_url = f"http://127.0.0.1:{cfg('cmdserver.port')}?suite={suite}"
    requests.get(start_url)

    wait_cnt = 0

    while True:
        try:
            requests.get(f"http://127.0.0.1:{cfg('docker.http_port')}")
            break
        except Exception as e:
            wait_cnt += 1
            if wait_cnt < 10:
                banner(f"WAITING FOR THE CONTAINER: {e.__class__.__name__}")
                time.sleep(5)
            else:
                banner('GIVING UP')
                sys.exit(1)

    exec_suite(suite, opts)


def run_batch(suites, opts):
    if suites:
        suites = [s.strip() for s in suites.split(',')]
    else:
        suites = []
        base = cfg('paths.app_root') + '/' + SUITE_BASE
        for p in os.listdir(base):
            if os.path.isdir(base + '/' + p):
                suites.append(p)

    for suite in sorted(suites):
        run_suite(suite, opts)


def start_postgres():
    # see https://hub.docker.com/r/kartoza/postgis/ for details

    opts = [
        f"--rm",
        f"--detach",
        f"--publish {cfg('postgres.port')}:5432",
        f"--env POSTGRES_USER={cfg('postgres.user')}",
        f"--env POSTGRES_PASS={cfg('postgres.password')}",
        f"--env POSTGRES_DB={cfg('postgres.database')}",
        f"--name {cfg('postgres.container_name')}",
    ]

    run(f"docker kill {cfg('postgres.container_name')}")
    time.sleep(2)
    run(f"docker rm --force {cfg('postgres.container_name')}")

    image_name = 'kartoza/postgis:12.0'

    run(f"docker run {' '.join(opts)} {image_name}", wait=False)


class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        params = {}
        if '?' in self.path:
            params = dict(urllib.parse.parse_qsl(self.path.split('?')[1]))

        if 'suite' in params:
            start_container_for_suite(params['suite'], ['-i'])

        if 'exit' in params:
            banner('STOPPING DOCKER SERVER')
            sys.exit(0)

        self.send_response(200)
        self.send_header("Content-type", 'text/plain')
        self.send_header("Content-Length", 0)
        self.end_headers()


def start_server():
    try:
        # signal the server to exit
        stop_url = f"http://127.0.0.1:{cfg('cmdserver.port')}?exit=1"
        requests.get(stop_url)
    except:
        pass

    atexit.register(stop_container)

    server_address = (cfg('cmdserver.host'), cfg('cmdserver.port'))

    HTTPRequestHandler.protocol_version = 'HTTP/1.1'
    httpd = http.server.HTTPServer(server_address, HTTPRequestHandler)

    sa = httpd.socket.getsockname()
    banner(f'STARTED DOCKER SERVER ON {sa[0]}:{sa[1]}')
    httpd.serve_forever()


def main():
    global CONFIG

    usage = """
    
        cmd.py server --config CONFIG
    
            start a simple HTTP server which run gws containers on demand: curl localhost:port?suite=SUITE
    
        cmd.py run --config CONFIG --suite SUITE --opts 'OPTS'
    
            tell the server to start a suite container and runs pytest on that suite
    
        cmd.py exec --config CONFIG --suite SUITE --opts 'OPTS'
    
            run pytest on a suite, assuming its container is started
    
        cmd.py batch --config CONFIG --suites SUITES
    
            run multiple/all suites
    
        cmd.py container --config CONFIG --suite SUITE 
    
            start a suite container interactively
        
        cmd.py postgres --config CONFIG
    
            start a postgres container
    """

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('cmd', help='command')
    parser.add_argument('--config', dest='config', help='configuration file path')
    parser.add_argument('--suite', dest='suite', help='suite name')
    parser.add_argument('--opts', dest='opts', help='pytest options')

    args = parser.parse_args()

    with open(args.config) as fp:
        CONFIG = json.load(fp)

    if args.cmd == 'server':
        start_server()
        return

    if args.cmd in ('run', 'r'):
        run_suite(args.suite, args.opts)
        return

    if args.cmd == 'batch':
        run_batch(args.suite, args.opts)
        return

    if args.cmd in ('exec', 'e'):
        exec_suite(args.suite, args.opts)
        return

    if args.cmd == 'container':
        start_container_for_suite(args.suite, ['-i'])
        return

    if args.cmd == 'postgres':
        start_postgres()
        return


if __name__ == '__main__':
    main()
