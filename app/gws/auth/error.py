import gws


class Error(gws.Error):
    """Generic autorization error"""
    pass


class LoginNotFound(Error):
    """The login name is not found"""
    pass


class WrongPassword(Error):
    """The login is found, but the password is wrong"""
    pass


class LoginFailed(Error):
    """Cannot login, provider error"""
    pass


class AccessDenied(Error):
    """Can login, but the acccess is denied"""
    pass
