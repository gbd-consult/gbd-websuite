"""Regex validator for strings.

Validates if the string matches regex. Uses ``re.search``,
that is, the start anchor must be included if necessary.
"""

import re

import gws
import gws.base.model.validator

gws.ext.new.modelValidator('regex')


class Config(gws.base.model.validator.Config):
    regex: gws.Regex


class Object(gws.base.model.validator.Object):
    regex: str

    def configure(self):
        self.regex = self.cfg('regex')

    def validate(self, field, feature, mc):
        val = feature.attributes.get(field.name)
        if not isinstance(val, str):
            return False
        m = re.search(self.regex, val)
        return m is not None
