import json
import os

import gws
import gws.types.spec


def load(lang=None):
    path = f'{gws.APP_DIR}/spec/gen/spec.json'

    if lang and lang != 'en':
        p = f'{gws.APP_DIR}/spec/gen/{lang}.spec.json'
        if os.path.exists(p):
            path = p

    with open(path, encoding='utf8') as fp:
        return json.load(fp)


def validator() -> gws.types.spec.Validator:
    def init():
        return gws.types.spec.Validator(load())
    return gws.get_global('spec_validator', init)
