"""Authentication providers.

An authentication provider checks the credentials passed by an authentication
method and returns a user with roles. Providers are configured in
``auth.providers``. The authentication manager tries the providers that allow
the method in the configured order and takes the first user found.

Subpackages
-----------

- ``file`` - users from a local JSON file with hashed passwords.
- ``ldap`` - users from an LDAP or Active Directory server, with roles
  assigned by LDAP filters and group membership.

Example::

    auth.providers+ {
        type "file"
        path "/data/users.json"
    }
"""
