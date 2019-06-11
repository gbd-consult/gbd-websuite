import getpass

import gws.config.loader
import gws.auth.api
import gws.auth.error
import gws.auth.session
import gws.tools.clihelpers
import gws.tools.date as dt

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
    """Logout all users and remove all active sessions"""

    man = gws.auth.session.Manager()
    man.delete_all()


def sessions():
    """Print currently active sessions"""

    gws.config.loader.load()
    man = gws.auth.session.Manager()

    rs = [{
        'user': r['user_uid'],
        'login': dt.to_iso(dt.from_timestamp(r['created'])),
        'activity': dt.to_iso(dt.from_timestamp(r['updated'])),
        'duration': r['updated'] - r['created']
    } for r in man.get_all()]

    print(f'{len(rs)} active sessions\n')
    print(gws.tools.clihelpers.text_table(rs, header=('user', 'login', 'activity', 'duration')))
    print('\n')
