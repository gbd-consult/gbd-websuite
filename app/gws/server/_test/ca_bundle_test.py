import functools
import http.server
import json
import os
import ssl
import subprocess
import threading

import gws
import gws.test.util as u

_ENV_VARS = ['SSL_CERT_FILE', 'CURL_CA_BUNDLE', 'LDAPTLS_CACERT']

_SCRIPT = '''
import json, os, subprocess, sys
import gws.lib.net
url = sys.argv[1]
print(json.dumps(dict(
    env={k: os.environ.get(k) for k in sys.argv[2:]},
    request_ok=gws.lib.net.http_request(url).ok,
    gdal=subprocess.run(f'gdalinfo /vsicurl/{url}ca.pem 2>&1', shell=True, capture_output=True, text=True).stdout,
)))
'''


@u.fixture(scope='module')
def ca(tmp_path_factory):
    d = tmp_path_factory.mktemp('ca')

    u.exec(f'''
        cd {d}
        openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.pem -days 2 -subj "/CN=ca_1" \\
            -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign"
        openssl req -newkey rsa:2048 -nodes -keyout srv.key -out srv.csr -subj "/CN=localhost"
        printf "subjectAltName=DNS:localhost" > ext.txt
        openssl x509 -req -in srv.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out srv.pem -days 2 -extfile ext.txt
    ''')
    gws.u.write_file(f'{d}/script.py', _SCRIPT)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(f'{d}/srv.pem', f'{d}/srv.key')
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    srv = http.server.HTTPServer(('localhost', 0), handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    yield gws.Data(dir=str(d), url=f'https://localhost:{srv.server_address[1]}/')

    srv.shutdown()


def _run(ca, extra_bundle):
    env = {k: v for k, v in os.environ.items() if k not in _ENV_VARS and k != 'GWS_EXTRA_CA_BUNDLE'}
    if extra_bundle:
        env['GWS_EXTRA_CA_BUNDLE'] = extra_bundle
    return subprocess.run(
        [f'{gws.c.APP_DIR}/bin/gws', '-p', f'{ca.dir}/script.py', ca.url, *_ENV_VARS],
        env=env,
        capture_output=True,
        text=True,
    )


def _result(p):
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_with_extra_bundle(ca):
    res = _result(_run(ca, f'{ca.dir}/ca.pem'))
    bundle_path = f'{gws.c.VAR_DIR}/config/ca-bundle.crt'

    assert res['env'] == {k: bundle_path for k in _ENV_VARS}
    bundle = gws.u.read_file(bundle_path)
    assert bundle.startswith(gws.u.read_file('/etc/ssl/certs/ca-certificates.crt'))
    assert bundle.endswith(gws.u.read_file(f'{ca.dir}/ca.pem'))
    assert res['request_ok']
    assert 'gdalinfo failed' in res['gdal'] and 'SSL' not in res['gdal']


def test_without_extra_bundle(ca):
    res = _result(_run(ca, ''))

    assert res['env']['SSL_CERT_FILE'] is None
    assert res['env']['LDAPTLS_CACERT'] is None
    assert not res['request_ok']
    assert 'SSL certificate verification failed' in res['gdal']


def test_missing_extra_bundle_fails(ca):
    p = _run(ca, f'{ca.dir}/missing_1.pem')
    assert p.returncode != 0
    assert 'GWS_EXTRA_CA_BUNDLE' in p.stderr
