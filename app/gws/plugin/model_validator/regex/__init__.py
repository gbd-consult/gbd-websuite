"""Regular expression validator for strings.

Checks that the field value is a string that matches a regular expression.
The match uses ``re.search``, so the expression can match anywhere in the
value; add ``^`` and ``$`` anchors to match the whole value. A value that is
not a string fails the check.

Example::

    fields+ {
        name "zip"
        type "text"
        validators+ { type "regex" regex "^[0-9]{5}$" }
    }
"""

import re

import gws
import gws.base.model.validator


@gws.ext.config.modelValidator('regex')
class Config(gws.base.model.validator.Config):
    """Checks that a value matches a regular expression."""

    regex: gws.Regex
    """Regular expression, matched anywhere in the value unless anchored."""


@gws.ext.object.modelValidator('regex')
class Object(gws.base.model.validator.Object):
    """Regular expression validator object."""

    regex: str
    """The regular expression."""

    def configure(self):
        self.regex = self.cfg('regex')

    def validate(self, field, feature, mc):
        val = feature.attributes.get(field.name)
        if not isinstance(val, str):
            return False
        m = re.search(self.regex, val)
        return m is not None
