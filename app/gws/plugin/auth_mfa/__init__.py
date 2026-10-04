"""Multi-factor authentication adapters.

A multi-factor adapter adds a second login step after a provider has
authenticated the user. Adapters are configured in ``auth.mfa``. A user is
assigned to an adapter by the ``mfaUid`` attribute of the user record, which
must match the ``uid`` of the adapter. The second step is run by the web
authentication method (``gws.plugin.auth_method.web``). The transaction life
cycle (life time, verification attempts, restarts) and the TOTP helpers are
implemented in the base class ``gws.base.auth.mfa``.

Subpackages
-----------

- ``email`` - sends a one-time code to the email address of the user.
- ``totp`` - checks time-based one-time passwords from an authenticator app,
  using the ``mfaSecret`` of the user.

Example::

    auth.mfa+ {
        type "totp"
        uid "AUTH_MFA_TOTP"
    }

A user record of the ``file`` provider that uses this adapter::

    {
        "login": "user_1",
        "password": "...",
        "mfaUid": "AUTH_MFA_TOTP",
        "mfaSecret": "..."
    }
"""
