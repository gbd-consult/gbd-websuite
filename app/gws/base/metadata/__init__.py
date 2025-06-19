"""OWS and ISO-19115 metadata.

The metadata structure as defined here is based on:

- ISO 19115:2003 Geographic information -- Metadata
- ISO 19139 - Geographic Information - Metadata - XML Schema Implementation
- OGC Web Services Common Standard OGC 06-121r9
- Web Map Server Implementation Specification OGC 06-042
- Web Feature Service 2.0 Interface Standard OGC 09-025r1
"""

from .core import (
    Config,
    Props, 
    props,
    from_args,
    from_config,
    from_dict,
    from_props,
    update,
    new,
    keyword_groups,
    KeywordGroup,
)

from . import inspire, iso
