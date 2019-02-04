import getpass

from argh import arg

import gws.auth.api
import gws.auth.error
import gws.auth.session
import gws.config.loader

COMMAND = 'auth'


@arg('--login', help='login')
@arg('--password', help='password')
def test(login=None, password=None):
    """Interactively test a login"""

    gws.config.loader.load()
    num_tries = 1 if (login and password) else 999

    while num_tries > 0:
        num_tries -= 1

        login = login or input('Username: ')
        password = password or getpass.getpass('Password: ')

        try:
            user = gws.auth.api.authenticate_user(login, password)
        except gws.auth.error.WrongPassword:
            print('wrong password!')
            continue

        if user is None:
            print('login not found!')
            continue

        print('logged in!')
        print(f'User display name: {user.display_name}')
        print(f'Roles: {user.roles}')
        break


def clear():
    """Logout all users and remove all active sessions"""

    man = gws.auth.session.Manager()
    man.delete_all()
