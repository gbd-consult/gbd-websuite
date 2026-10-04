"""Authentication methods.

An authentication method defines how credentials and sessions are passed
between the client and the server. Methods are configured in ``auth.methods``.
The credentials a method extracts from a request are checked by the
authentication providers (``auth.providers``).

Subpackages
-----------

- ``basic`` - HTTP Basic authentication. The credentials are sent with every
  request in the ``Authorization`` header, and each request gets a transient
  session.
- ``token`` - a token in a configurable HTTP header. Each request gets a
  transient session.
- ``web`` - a login form and a session cookie. Includes the ``auth`` action,
  the client API for login, logout and multi-factor authentication, and an
  optional redirect of denied page requests to a login page.

Example::

    auth.methods+ {
        type "web"
        loginRedirect {
            pattern "^/project"
            target "/login"
        }
    }

    auth.methods+ { type "basic" }

    auth.providers+ { type "file" path "/data/users.json" }

    actions+ {
        type "auth"
        permissions.read "allow all"
    }
"""
