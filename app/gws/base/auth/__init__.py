"""Authentication and authorization.

This package provides the authorization manager, the base classes for the
authentication plugins and the user objects that check permissions.

Authentication is split into four kinds of pluggable objects:

- methods (``gws.AuthMethod``, plugins in ``gws.plugin.auth_method``, for example
  ``web``, ``basic``, ``token``) define how credentials and sessions reach the server,
  for example as a login form and a session cookie, or as HTTP basic auth.
- providers (``gws.AuthProvider``, for example ``file``, ``ldap``, ``postgres``)
  verify credentials and return users.
- multi-factor adapters (``gws.AuthMultiFactorAdapter``, for example ``email``,
  ``totp``) add a second verification step for users with an ``mfaUid``.
- the session manager (``gws.AuthSessionManager``, by default ``sqlite``)
  stores sessions between requests.

Submodules:

- ``manager``: the authorization manager (``root.app.authMgr``). It creates the
  methods, providers, adapters, session manager and the optional throttle, and,
  as the ``auth`` middleware, opens a session for every web request.
- ``method``: base class for authentication methods.
- ``provider``: base class for authentication providers.
- ``system_provider``: the built-in provider of the ``guest`` and ``system`` users.
- ``mfa``: base class for multi-factor adapters, with TOTP code generation and checking.
- ``session``: the session object.
- ``session_manager``: base class for session managers.
- ``throttle``: blocking of repeated failed login attempts, counted per address and,
  optionally, per login name, and stored in an sqlite file.
- ``user``: user classes, permission checks and conversion of provider records to users.
- ``cli``: the ``gws auth`` CLI commands to list and remove sessions.

Multi-factor authentication (handled by the ``web`` method) is used for users
with an ``mfaUid`` attribute, the uid of a configured adapter. Specific adapters
can require other user attributes, for example ``email`` or ``mfaSecret``. The
login starts a ``gws.AuthMultiFactorTransaction``, which is kept in the session
until it is verified, fails or expires. Some adapters can be restarted, for
example by sending a new code by email. A transaction fails when its life time
has elapsed or the number of verification attempts exceeds the limit. A restart
creates a new transaction with an increased restart count.

If no methods are configured, the ``web`` method is used. The ``system``
provider with the guest and system users is always added after the configured
providers. ``maxLifeTime`` of the session manager, if set, must not be less than
``lifeTime``.

The manager is registered as the ``auth`` middleware, depending on ``db``. On
each web request, it asks every method in turn to open a session. If no method
returns one, the request runs with the guest session. After the request, the
method of the session closes it, for example by setting the session cookie.
Setting a session value marks the session as changed. When a user logs in, the
manager passes the credentials to each provider that allows the method, until
one returns a user.

With a throttle configured, blocked login attempts raise
``gws.TooManyRequestsError`` and every attempt is registered with the throttle.
The login name is stored as a hash. The counts are kept in an sqlite table, so
they are shared between server processes. ``blockTime`` must be greater than
``windowTime``.

Users carry a set of roles. Every user has the role ``all``; logged-in users
also have ``user`` (or ``admin``), the guest user has ``guest``. A user can
always read itself. Otherwise, access to an object is decided by the first
permission entry whose role the user has, walking up the given context objects
and then the object's parents until a decision is found. Without a decision,
access is denied. The ``system`` user and admin users are allowed everything.

Example::

    auth {
        methods [
            { type web secure true }
        ]
        providers [
            { type file path "/data/users.json" }
        ]
        session { type sqlite lifeTime "30m" }
        throttle { maxAttemptsPerIp 5 }
    }

Example::

    user = req.user
    if user.can_write(layer):
        ...
    project = user.require_project(p.projectUid)
"""

from . import (
    manager,
    method,
    mfa,
    provider,
    session,
    session_manager,
    throttle,
    user,
)
