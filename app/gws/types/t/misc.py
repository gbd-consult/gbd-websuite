### Miscellaneous types.

from .base import List, Optional, DirPath
from .data import Config

class DocumentRootConfig(Config):
    """Base directory for assets"""

    dir: DirPath  #: directory path
    allowMime: Optional[List[str]]  #: allowed mime types
    denyMime: Optional[List[str]]  #: disallowed mime types (from the standard list)


