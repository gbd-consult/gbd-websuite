import gws


class Error(gws.Error):
    pass


class PasswordsDoNotMatch(Error):
    pass


class PasswordNotValidated(Error):
    pass


class MultipleEntriesFound(Error):
    pass


class WrongLogin(Error):
    pass


class WrongPassword(Error):
    pass


class InvalidMfaIndex(Error):
    pass


class NoEmail(Error):
    pass
