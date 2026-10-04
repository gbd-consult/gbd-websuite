"""Web authentication method.

Users log in with a login form in the client. On success, the method creates a
session in the session manager and sends its uid to the browser in an
``HttpOnly`` session cookie. On every request, the session is looked up by the
cookie. If the cookie refers to an unknown or expired session, the request gets
a placeholder "deleted" session for the guest user, and the cookie is removed
at the end of the request. For other sessions, the cookie is set and the
session is touched at the end of the request, if the response status is below
400.

If the authenticated user has an ``mfaUid``, the login needs a second step: the
method starts a transaction with the multi-factor adapter of that uid and keeps
it in a guest session until the user has entered a valid code.

If ``secure`` is set (the default), logins are only accepted over HTTPS and the
cookie is marked ``Secure``.

Submodules
----------

- ``core`` - the ``web`` authentication method: session cookies, login,
  logout, multi-factor verification and restart, and the redirect of denied
  page requests to a login page (``loginRedirect``). Also the request and
  response types of the login API.
- ``action`` - the ``auth`` action, the client API of the method:
  ``authCheck``, ``authLogin``, ``authLogout``, ``authMfaVerify`` and
  ``authMfaRestart``. The action requires a configured ``web`` method.
- ``js`` - the client part.

With ``loginRedirect``, denied GET requests (status ``401`` or ``403``) whose
URI matches ``pattern`` are redirected to ``target``. The original URI is
passed in the ``to`` parameter; the client can send it back with the login
request to return there after the login.

Example::

    auth.methods+ {
        type "web"
        cookieName "auth"
        loginRedirect {
            pattern "^/(demo|project)"
            target "/login"
        }
    }

    actions+ {
        type "auth"
        permissions.read "allow all"
    }
"""
