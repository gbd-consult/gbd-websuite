### Access rules and configs.

from .base import List, Enum, Optional
from ..data import Config


class AccessType(Enum):
    allow = 'allow'
    deny = 'deny'


class AccessRuleConfig(Config):
    """Access rights definition for authorization roles"""

    type: AccessType  #: access type (deny or allow)
    role: str  #: a roles to which this rule applies


#: alias:
Access = List[AccessRuleConfig]


class WithType(Config):
    type: str  #: object type


class WithTypeAndAccess(WithType):
    access: Optional[Access]  #: access rights
