### Access rules and configs.

from .base import List, Enum, Optional
from ..data import Config


class AccessType(Enum):
    allow = 'allow'
    deny = 'deny'


class AccessRuleConfig(Config):
    """Access rights definition for authorization roles"""

    type: AccessType  #: access type (deny or allow)
    role: str  #: a role to which this rule applies


class WithType(Config):
    type: str  #: object type


class WithAccess(Config):
    access: Optional[List[AccessRuleConfig]]  #: access rights


class WithTypeAndAccess(Config):
    type: str  #: object type
    access: Optional[List[AccessRuleConfig]]  #: access rights
