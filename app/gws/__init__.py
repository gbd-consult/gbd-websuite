"""Core gws module"""

from .core import ext
from .core import log
from .core.const import *
from .core.data import Data, is_data_object
from .core.debug import p, time_start, time_end
from .core.error import Error, ConfigurationError
from .core.tree import Object, Node, Root, create_root_object, class_name, props
from .core.types import *
from .core.util import *
