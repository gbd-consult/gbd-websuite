import getpass
import gws.tools.password

COMMAND = 'auth'


def passwd():
    """Encode a password for the authorization file"""

    p1 = getpass.getpass('Password: ')
    p2 = getpass.getpass('Repeat  : ')
    if p1 == p2:
        print(gws.tools.password.encode(p1))
    else:
        print('passwords do not match')
