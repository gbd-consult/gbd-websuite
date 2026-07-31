"""Web requests.

ref: https://werkzeug.palletsprojects.com/en/3.0.x/test/
"""

from typing import cast

import re

import werkzeug.test

import gws
import gws.base.web.wsgi_app


class TestResponse(werkzeug.test.TestResponse):
    cookies: dict[str, werkzeug.test.Cookie]


def get(root, url, **kwargs) -> TestResponse:
    url = re.sub(r'\s+', '', url.strip())
    url = '/' + url.strip('/')
    return _wz_request(root, method='GET', path=url, **kwargs)


def post(root, url, **kwargs) -> TestResponse:
    url = re.sub(r'\s+', '', url.strip())
    url = '/' + url.strip('/')
    return _wz_request(root, method='POST', path=url, **kwargs)


def api(root, cmd, request=None, **kwargs) -> TestResponse:
    path = gws.c.SERVER_ENDPOINT
    if cmd:
        path += '/' + cmd
    return _wz_request(root, method='POST', path=path, json=request or {}, **kwargs)


def _wz_request(root, **kwargs):
    client = werkzeug.test.Client(gws.base.web.wsgi_app.make_application(root))

    cookies = cast(list[werkzeug.test.Cookie], kwargs.pop('cookies', []))
    for c in cookies:
        client.set_cookie(
            key=c.key,
            value=c.value,
            max_age=c.max_age,
            expires=c.expires,
            path=c.path,
            domain=c.domain,
            secure=c.secure,
            httponly=c.http_only,
        )

    res = client.open(**kwargs)

    # for some reason, responses do not include cookies, work around this
    res.cookies = {c.key: c for c in (client._cookies or {}).values()}
    return res
