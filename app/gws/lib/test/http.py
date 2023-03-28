import http.cookies

import werkzeug.test
import werkzeug.wrappers

import gws
import gws.base.web.wsgi_app
import gws.lib.jsonx
import gws.lib.net


def local_request(url, **kwargs):
    """Perform a get request to the local server."""

    return gws.lib.net.http_request('http://127.0.0.1' + '/' + url, **kwargs)


class ClientCmdResponse(gws.Data):
    status: int
    json: dict
    cookies: dict
    response: werkzeug.wrappers.Response


def client_cmd_request(cmd, params, cookies=None, headers=None) -> ClientCmdResponse:
    gws.log.debug(f'TEST:client_cmd_request {cmd}')

    client = _prepare_client(cookies)

    resp = client.open(
        method='POST',
        path='/_/' + cmd,
        data=gws.lib.jsonx.to_string({'params': params}),
        content_type='application/json',
        headers=headers,
    )

    js = None
    try:
        js = gws.lib.jsonx.from_string(resp.data)
    except:
        pass

    cookie_headers = ';'.join(v for k, v in resp.headers if k == 'Set-Cookie')
    response_cookies = {}

    mor: http.cookies.Morsel
    for k, mor in http.cookies.SimpleCookie(cookie_headers).items():
        response_cookies[k] = dict(mor)
        response_cookies[k]['value'] = mor.value

    return ClientCmdResponse(
        status=resp.status_code,
        json=js,
        cookies=response_cookies,
        response=resp,
    )


def _prepare_client(cookies):
    client = werkzeug.test.Client(
        gws.base.web.web_app.application,
        werkzeug.wrappers.Response)

    if cookies:
        for k, v in cookies.items():
            if not v:
                client.delete_cookie('localhost', k)
            elif isinstance(v, str):
                client.set_cookie('localhost', k, v)
            else:
                client.set_cookie('localhost', k, **v)

    return client
