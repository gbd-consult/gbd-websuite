import getpass
import gws.config.loader
import gws.auth.api
import gws.auth.error
import gws.auth.session

COMMAND = 'auth'


def test():
    """Interactively test a login"""

    gws.config.loader.load()

    while True:
        login = input('Username: ')
        password = getpass.getpass('Password: ')

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
    """Logo ut all users and remove all active sessions"""

    man = gws.auth.session.Manager()
    man.delete_all()
