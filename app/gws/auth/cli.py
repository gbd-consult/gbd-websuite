import getpass

from argh import arg

import gws.auth.api
import gws.auth.error
import gws.auth.session
import gws.config.loader
import gws.tools.clihelpers
import gws.tools.date as dt
import gws.tools.password

COMMAND = 'auth'


@arg('--login', help='login')
@arg('--password', help='password')
def test(login=None, password=None):
    """Interactively test a login"""

    gws.config.loader.load()

    login = login or input('Username: ')
    password = password or getpass.getpass('Password: ')

    try:
        user = gws.auth.api.authenticate_user(login, password)
    except gws.auth.error.WrongPassword:
        print('wrong password!')
        return

    if user is None:
        print('login not found!')
        return

    print('logged in!')
    print(f'User display name: {user.display_name}')
    print(f'Roles: {user.roles}')


def clear():
    """Logout all users and remove all active sessions"""

    man = gws.auth.session.Manager()
    man.delete_all()


@arg('--path', help='where to store the password')
def passwd(path=None):
    """Encode a password for the authorization file"""

    while True:
        p1 = getpass.getpass('Password: ')
        p2 = getpass.getpass('Repeat  : ')

        if p1 != p2:
            print('passwords do not match')
            continue

        p = gws.tools.password.encode(p1)
        if path:
            with open(path, 'wt') as fp:
                fp.write(p + '\n')
        else:
            print(p)
        break


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
