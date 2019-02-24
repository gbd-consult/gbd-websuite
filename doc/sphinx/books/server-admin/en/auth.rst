Authorization
=============

GBD WebSuite authorization is role-based, with pluggable authorization providers. Each user in the system has a list of accociated roles. The access to specific objects is based on user's roles.


Access rules
------------

Some types of the objects in the configuration can have ``access`` configurations attached to them:

- main application
- server action
- project
- map
- layer

Additionally, some actions define internal ``access`` blocks for specific commands.

An ``access`` block is a list of ``AccessRule`` objects. Each ``AccessRule`` contains

- the ``type`` of the rule - "allow" or "deny",
- the ``role`` name this rule applies to

When the user requests an object, the server checks the access rules defined in this object against the list of the user's roles. If any role has been found, the access is granted or denied, depending on the rule, otherwise the parent object is checked, until the root of the object hierarchy is reached, in which case the access is denied.

Roles
-----

A ``role`` is just an identifier and can be freely choosen. There are a few predefined roles that have special meaning in GBD WebSuite:

TABLE
   *guest* ~ Not logged-in user
   *user* ~ Any logged-in user
   *all* ~ All users, logged-in and guests
   *admin* ~ Administrator. Users that have this role are automatically granted access to all resources
/TABLE


Authorization strategies
------------------------

Since access rules are inherited, the first thing you have to configure is the root ``access`` block. If your projects are mostly public (or when you don't need any authorization at all), you can grant access to "all" in the topmost config ::

    ## in the main config:

    "access": [
        {
            "type": "allow",
            "role": "all"
        }
    ]

Now, if you need to restrict access to some object, e.g. a project, you need two access rules: one to allow a specific role, and one to deny "all" ::

    ## in the project config:

    "access": [
        {
            "type": "allow",
            "role": "member"
        },
        {
            "type": "deny",
            "role": "all"
        }
    ]

On the other side, if most of your projects require a login, it's easier to start with a "deny all" rule ::

    ## in the main config:

    "access": [
        {
            "type": "deny",
            "role": "all"
        }
    ]

and then explicitly allow access to specific objects ::

    # in the project config:

    "access": [
        {
            "type": "allow",
            "role": "member"
        }
    ]


Authorization providers
-----------------------


When the user logs in, their credentials are passed to all configured providers in turn. If some provider accepts the credentials, it is supposed to return a list of roles for this user.


file
~~~~

The file provider uses a simple json file to store authorization data. The json is just an array of "user" objects ::


    [
        {
            "login": "user login",
            "password": "sha512 encoded password",
            "name": "display name for the user",
            "roles": [ "role1", "role2", ...]
        },
        {
            ...
        }
    }

The name and the location of the file is up to you, just specify its absolute path in the configuration. To generate the encoded password, use the ``auth passwd`` command.

ldap
~~~~

The ldap provider can authorize users against an ActiveDirectory or an OpenLDAP server. You should configure at least an URL of the server and a set of rules to map LDAP filters to GWS role names. Here's an example configuration using `the test LDAP server provided by forumsys.com <http://www.forumsys.com/tutorials/integration-how-to/ldap/online-ldap-test-server>`_ ::


    {
        "type": "ldap",

        ## the URL format is  "ldap://host:port/baseDN?searchAttribute":

        "url": "ldap://ldap.forumsys.com:389/dc=example,dc=com?uid",

        ## credentials to bind to the server:

        "bindDN": "cn=read-only-admin,dc=example,dc=com",
        "bindPassword": "password",

        ## map filters to roles:

        "users": [

            ## LDAP user "newton" possesses the GWS role "moderator" and "member":

            {
                "matches": "(&(cn=newton))",
                "roles": ["moderator", "member"]
            },

            ## all members of the LDAP group "mathematicians" possess the GWS role "member":

            {
                "memberOf": "(&(ou=mathematicians))",
                "roles": ["member"]
            }
        ]
    }
