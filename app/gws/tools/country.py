"""Country/language codes and tools."""

import pycountry

_cache = {
    'en': 'eng',
    'de': 'ger',
    'EN': 'eng',
    'DE': 'ger',
}

def bibliographic_name(language=''):
    if language:
        if language not in _cache:
            _cache[language] = pycountry.languages.get(alpha_2=language.lower()).bibliographic
        return _cache[language]
