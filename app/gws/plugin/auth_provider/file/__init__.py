"""Authentication provider for users stored in a JSON file.

The file contains a list of user records (dicts). Each record must contain
``login`` and ``password``, the password hashed with
``gws.lib.password.encode``. ``name`` is used as the display name. Other
fields, for example ``roles``, ``email`` or ``mfaUid``, are passed to
``gws.base.auth.user.from_record`` and become properties of the user.

The provider accepts ``username`` and ``password`` credentials. If the login
is not found, a dummy hash is checked anyway, so that the response time does
not reveal whether a login exists. The file is read once, at configuration
time.

The command ``gws auth password`` asks for a password and prints its hash for
the file.

Example::

    auth.providers+ {
        type "file"
        path "/data/users.json"
    }

with ``/data/users.json``::

    [
        {
            "login": "user_1",
            "password": "<hash>",
            "name": "User 1",
            "roles": ["editor"]
        }
    ]
"""

import getpass

import gws
import gws.base.auth
import gws.lib.jsonx
import gws.lib.password


@gws.ext.config.authProvider('file')
class Config(gws.base.auth.provider.Config):
    """Authentication against user records in a JSON file."""

    path: gws.FilePath
    """Path to the JSON file with user records."""


@gws.ext.object.authProvider('file')
class Object(gws.base.auth.provider.Object):
    """File authentication provider."""

    path: str
    """Path to the JSON file."""
    db: list[dict]
    """User records read from the file."""
    dummyPassword: str
    """Hash of a random password, checked when a login is not found."""

    def configure(self):
        self.path = self.cfg('path')
        self.db = gws.lib.jsonx.from_path(self.path)
        self.dummyPassword = gws.lib.password.encode(gws.u.random_string(32))

    def authenticate(self, method, credentials):
        username = credentials.get('username')
        password = credentials.get('password')
        if not username or not password:
            return

        found = [rec for rec in self.db if gws.lib.password.compare(username, rec['login'])]

        if len(found) > 1:
            raise gws.AuthenticationError(f'multiple entries for {username!r}')

        if not found:
            # verify against a dummy hash, so that the time spent here
            # does not reveal whether the login exists
            gws.lib.password.check(password, self.dummyPassword)
            return

        if not gws.lib.password.check(password, found[0]['password']):
            raise gws.AuthenticationError(f'wrong password for {username!r}')

        return self._make_user(found[0])

    def get_user(self, local_uid):
        for rec in self.db:
            if rec['login'] == local_uid:
                return self._make_user(rec)

    def _make_user(self, rec: dict):
        """Create a user from a file record."""
        user_rec = dict(rec)

        login = user_rec.pop('login', '')
        user_rec['localUid'] = user_rec['loginName'] = login
        user_rec['displayName'] = user_rec.pop('name', login)
        user_rec.pop('password', '')

        return gws.base.auth.user.from_record(self, user_rec)

    @gws.ext.command.cli('authPassword')
    def passwd(self, p: gws.EmptyRequest):
        """Ask for a password and print its hash for the users file."""

        while True:
            p1 = getpass.getpass('Password: ')
            p2 = getpass.getpass('Repeat  : ')

            if p1 != p2:
                print('passwords do not match')
                continue

            p = gws.lib.password.encode(p1)
            print(p)
            break
