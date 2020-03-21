import json
import os

import gws
import gws.core.spec


def load(lang=None) -> gws.core.spec.SpecValidator:
    path = f'{gws.APP_DIR}/spec/gen/spec.json'

    if lang and lang != 'en':
        p = f'{gws.APP_DIR}/spec/gen/{lang}.spec.json'
        if os.path.exists(p):
            path = p

    with open(path, encoding='utf8') as fp:
        return json.load(fp)


def validator() -> gws.core.spec.SpecValidator:
    def init():
        return gws.core.spec.SpecValidator(load())
    return gws.get_global('spec_validator', init)
