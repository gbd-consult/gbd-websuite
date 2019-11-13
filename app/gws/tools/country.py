"""Country/language codes and tools."""

import pycountry


def bibliographic_name(language=''):
    if language:
        return pycountry.languages.get(alpha_2=language.lower()).bibliographic
