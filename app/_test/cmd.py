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


def docker_run(image, opts, cmd=''):
    run(f"docker run {' '.join(opts)} {image} {cmd}", wait=False)


def start_suite_container(suite, extra_options=None):
    stop_suite_container()

    opts = [
        f"--add-host=mainhost:{cfg('docker.host_ip')}",
        f"--env GWS_CONFIG=/data/config.cx",
        f"--env GWS_TMP_DIR=/gws-var/tmp",
        f"--mount type=bind,src={cfg('paths.app_root')},dst=/gws-app",
        f"--mount type=bind,src={cfg('paths.app_root')}/_test/common,dst=/common",
        f"--mount type=bind,src={cfg('paths.app_root')}/_test/suite/{suite}/data,dst=/data",
        f"--mount type=bind,src={cfg('paths.var_root')},dst=/gws-var",
        f"--name {cfg('docker.container_name')}",
        f"--publish 0.0.0.0:{cfg('docker.http_port')}:80",
    ]

    opts += cfg('docker.options')

    if extra_options:
        opts += extra_options

    # write a copy of our config so that the container can read it

    with open(cfg('paths.var_root') + '/test.config.json', 'w') as fp:
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

    with open(cfg('paths.var_root') + '/start.sh', 'w') as fp:
        fp.write(start_script.strip())

    # ready to go

    banner('CONTAINER START')
    docker_run(cfg('docker.image_name'), opts, 'bash /gws-var/start.sh')


def stop_suite_container():
    banner('CONTAINER STOP')
    run(f"docker kill --signal SIGINT {cfg('docker.container_name')}")
    run(f"docker rm --force {cfg('docker.container_name')}")
    run(f"rm -fr {cfg('paths.var_root')}/*")


def start_postgres():
    # see https://hub.docker.com/r/kartoza/postgis/ for details

    stop_postgres()

    opts = [
        f"--detach",
        f"--env POSTGRES_DB={cfg('postgres.database')}",
        f"--env POSTGRES_PASS={cfg('postgres.password')}",
        f"--env POSTGRES_USER={cfg('postgres.user')}",
        f"--name {cfg('postgres.container_name')}",
        f"--publish {cfg('postgres.port')}:5432",
    ]

    image_name = 'kartoza/postgis:12.0'

    banner('STARTING POSTGRES...')
    docker_run(image_name, opts)


def stop_postgres():
    banner('STOPPING POSTGRES...')
    run(f"docker kill {cfg('postgres.container_name')}")
    run(f"docker rm --force {cfg('postgres.container_name')}")


def patch_qgis_service_urls():
    port = cfg('qgis.port')
    base = f"{cfg('paths.app_root')}/_test/common/qgis"

    for p in os.listdir(base):
        if p.endswith('.qgs'):
            with open(base + '/' + p) as fp:
                src = fp.read()

            urls = f"""
                <WMSUrl type="QString">http://mainhost:{port}?MAP=/qgis/{p}</WMSUrl>
                <WMTSUrl type="QString">http://mainhost:{port}?MAP=/qgis/{p}</WMTSUrl>
                <WFSUrl type="QString">http://mainhost:{port}?MAP=/qgis/{p}</WFSUrl>
            """

            src = re.sub(r'<(WMS|WMTS|WFS)Url.+\n', '', src)
            src = re.sub(r'</properties>', urls + '</properties>', src)

            with open(base + '/' + p, 'w') as fp:
                fp.write(src)


def start_qgis():
    # we use our own image to expose the qgis server which will provide us with test OWS services
    # port 4000 (our qgis port) is exposed to the host
    # the server serves projects from _test/common/qgis
    # qgs project files must be patched to provide correct callback urls (host:qgis-port)

    stop_qgis()

    opts = [
        f"--add-host=mainhost:{cfg('docker.host_ip')}",
        f"--detach",
        f"--env GWS_CONFIG=/qgis/config.cx",
        f"--env GWS_TMP_DIR=/gws-var/tmp",
        f"--mount type=bind,src={cfg('paths.app_root')},dst=/gws-app",
        f"--mount type=bind,src={cfg('paths.app_root')}/_test/common/qgis,dst=/qgis",
        f"--mount type=bind,src={cfg('paths.qgis_var_root')},dst=/gws-var",
        f"--name {cfg('qgis.container_name')}",
        f"--publish 0.0.0.0:{cfg('qgis.port')}:4000",
    ]

    banner('STARTING QGIS...')
    docker_run(cfg('docker.image_name'), opts, 'bash /gws-app/bin/gws server start -v')


def stop_qgis():
    banner('STOPPING QGIS...')
    run(f"docker kill {cfg('qgis.container_name')}")
    run(f"docker rm --force {cfg('qgis.container_name')}")


def stop_all():
    stop_suite_container()
    stop_postgres()
    stop_qgis()


##

def start_container_for_suite(suite, extra_options=None):
    run(f"make -C {cfg('paths.app_root')}/.. spec")
    patch_qgis_service_urls()
    start_suite_container(suite, extra_options)


def exec_suite(suite, opts):
    run(f"docker exec {cfg('docker.container_name')} pytest /gws-app/_test/suite/{suite} {opts or ''}")


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
        base = f"cfg('paths.app_root')/_test/suite"
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
        stop_url = f"http://127.0.0.1:{cfg('cmdserver.port')}?exit=1"
        requests.get(stop_url)
    except:
        pass

    atexit.register(stop_all)

    start_postgres()
    start_qgis()

    server_address = (cfg('cmdserver.host'), cfg('cmdserver.port'))

    HTTPRequestHandler.protocol_version = 'HTTP/1.1'
    httpd = http.server.HTTPServer(server_address, HTTPRequestHandler)

    sa = httpd.socket.getsockname()
    banner(f'STARTED CMD SERVER ON {sa[0]}:{sa[1]}')
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
        
        cmd.py qgis --config CONFIG
    
            start a qgis container
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
        start_cmd_server()
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

    if args.cmd == 'qgis':
        start_qgis()
        return


if __name__ == '__main__':
    main()
