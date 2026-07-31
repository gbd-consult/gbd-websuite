"""Mock server client."""

import requests

from . import options


def add(text):
    post('__add', data=text)


def set(text):
    post('__set', data=text)


def reset():
    post('__del')


def post(verb, data=''):
    requests.post(url(verb), data=data)


def url(path=''):
    h = options.option('service.mockserver.host')
    p = options.option('service.mockserver.port')
    u = f'http://{h}:{p}'
    if path:
        u += '/' + path
    return u
