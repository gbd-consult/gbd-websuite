import gws


class Category:
    onboarding = 'onboarding'
    resetPassword = 'resetPassword'


class Status(gws.Enum):
    new = 0
    onboarding = 1
    active = 10


class Columns:
    username = 'username'
    email = 'email'
    status = 'status'
    password = 'password'
    mfaUid = 'mfauid'
    mfaSecret = 'mfasecret'
    tc = 'tc'
    tcTime = 'tctime'
    tcCategory = 'tccategory'
