"""Utilities to manipulate metadata"""

import gws
import gws.tools.country
import gws.types as t

def read(m: t.MetaData) -> t.MetaData:
    if not m:
        return t.MetaData()
    if m.get('language'):
        m.language3 = gws.tools.country.bibliographic_name(language=m.language)
    return m
