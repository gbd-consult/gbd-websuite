import json
import os

import gws
import gws.types.spec


def load(kind, lang=None):
    path = f'{gws.APP_DIR}/spec/gen/{kind}.spec.json'

    if lang and lang != 'en':
        p = f'{gws.APP_DIR}/spec/gen/{lang}.{kind}.spec.json'
        if os.path.exists(p):
            path = p

    with open(path, encoding='utf8') as fp:
        return json.load(fp)


def config_validator():
    def init():
        s = load('config')
        return gws.types.spec.Validator(s['types'], strict=True)

    return gws.get_global('config_validator', init)


def action_validator():
    def init():
        s = load('api')
        return gws.types.spec.Validator(s['types'], strict=True)

    return gws.get_global('action_validator', init)


def action_commands():
    s = load('api')
    return s['methods']
