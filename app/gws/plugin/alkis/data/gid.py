"""GeoInfoDok schema of the ALKIS source data.

Re-exports ``geo_info_dok.gid6`` or ``geo_info_dok.gid7``, depending on
``GWS_ALKIS_GID_VERSION`` (default ``7``).
"""

import gws

if gws.env.GWS_ALKIS_GID_VERSION == '6':
    from .geo_info_dok.gid6 import *
else:
    from .geo_info_dok.gid7 import *
