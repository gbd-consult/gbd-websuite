"""OWS and ISO 19115 metadata.

Metadata describes an object (the application, a project, a layer, an OWS
service) for the client, for OWS capabilities documents and for CSW catalog
records. It is held in a flat ``gws.Metadata`` data object. The structure is
based on:

- ISO 19115:2003 Geographic information -- Metadata
- ISO 19139 - Geographic Information - Metadata - XML Schema Implementation
- OGC Web Services Common Standard OGC 06-121r9
- Web Map Server Implementation Specification OGC 06-042
- Web Feature Service 2.0 Interface Standard OGC 09-025r1

Submodules:

- ``core``: the metadata ``Config`` and ``Props`` and the functions that create,
  merge and convert metadata objects. ``new`` creates an empty object;
  ``from_dict``, ``from_config``, ``from_props`` and ``from_args`` create one
  from other sources; ``update`` merges values into an existing object;
  ``normalize`` returns a cleaned copy; ``props`` converts metadata to client
  props; ``keyword_groups`` groups keywords by vocabulary for OWS output.
  Merging is done key by key: later sources override earlier ones, except for
  ``keywords`` and ``isoTopicCategories``, which are combined. Date values are
  parsed to ``datetime`` objects, and the derived language codes and INSPIRE
  theme names are filled in on every update.
- ``inspire``: INSPIRE code lists as enums and the names and definitions of
  INSPIRE data themes.
- ``iso``: ISO 19115 code lists as enums.
- ``inspire_generate``: a script that downloads the INSPIRE theme register and
  regenerates the theme part of ``inspire``. It is not imported.

Example::

    metadata {
        title "Roads"
        abstract "Road network of the city"
        keywords [ "roads" "transport" ]
        language "en"
        inspireTheme "tn"
        isoTopicCategories [ "transportation" ]
    }

Usage in Python::

    md = gws.base.metadata.from_args(app_metadata, self.cfg('metadata'))
    md = gws.base.metadata.update(md, title='Roads')
    for kg in gws.base.metadata.keyword_groups(md):
        print(kg.codeSpace, kg.keywords)
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
    normalize,
    new,
    keyword_groups,
    KeywordGroup,
)

from . import inspire, iso
