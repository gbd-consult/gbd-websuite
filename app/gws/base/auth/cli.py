import gws


# @gws.ext.Object('cli.auth')
# class Object(gws.Object):
#
#     @gws.ext.command()
#     def test(self, login=None, password=None):
#         pass
#
# import getpass
#
# from argh import arg
#
# import gws.base.auth
# import gws.base.auth.session
# import gws.config.loader
# import gws.lib.clihelpers
# import gws.lib.date as dt
# import gws.lib.password
#
# COMMAND = 'auth'
#
#
# @arg('--login', help='login')
# @arg('--password', help='password')
# def test(login=None, password=None):
#     """Interactively test a login"""
#
#     login = login or input('Username: ')
#     password = password or getpass.getpass('Password: ')
#
#     auth = gws.config.loader.load().application.auth
#
#     try:
#         user = auth.authenticate(auth.get_method('web'), login, password)
#     except gws.base.auth.error.WrongPassword:
#         print('wrong password!')
#         return
#
#     if user is None:
#         print('login not found!')
#         return
#
#     print('logged in!')
#     print(f'User display name: {user.display_name}')
#     print(f'Roles: {user.roles}')
#
#
# @arg('--path', help='where to store the password')
# def passwd(path=None):
#     """Encode a password for the authorization file"""
#
#     while True:
#         p1 = getpass.getpass('Password: ')
#         p2 = getpass.getpass('Repeat  : ')
#
#         if p1 != p2:
#             print('passwords do not match')
#             continue
#
#         p = gws.lib.password.encode(p1)
#         if path:
#             with open(path, 'wt') as fp:
#                 fp.write(p + '\n')
#         else:
#             print(p)
#         break
#
#
# def clear():
#     """Logout all users and remove all active sessions"""
#
#     auth = gws.config.loader.load().application.auth
#     auth.delete_stored_sessions()
#
#
# def sessions():
#     """Print currently active sessions"""
#
#     auth = gws.config.loader.load().application.auth
#
#     rs = [{
#         'user': r['user_uid'],
#         'login': dt.to_iso_local(dt.from_timestamp(r['created'])),
#         'activity': dt.to_iso_local(dt.from_timestamp(r['updated'])),
#         'duration': r['updated'] - r['created']
#     } for r in auth.stored_session_records()]
#
#     print(f'{len(rs)} active sessions\n')
#     print(gws.lib.clihelpers.text_table(rs, header=('user', 'login', 'activity', 'duration')))
#     print('\n')
