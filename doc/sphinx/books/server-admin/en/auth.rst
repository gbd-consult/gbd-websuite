Authorization
=============

GBD WebSuite authorization is role-based, with pluggable authorization providers. When the user logs in, the user credentials are passed to all configured providers in turn. If a provider accepts the credentials, it returns a list of roles for this user.

When the user requests an object, the server checks if any of the user's roles has sufficient permissions to read the object, write to it (e.g. when editing) or execute it (e.g. a server action). If there are no explicit permissions on the object level, its parent object is checked and so on. To perform these checks, the server reads ``access`` rules of each object being requested.


Access rules
------------

``access`` is a list of ``AccessRule`` objects. Each ``AccessRule`` contains

- the ``type`` of the rule - "allow" or "deny",
- the ``mode`` - "read", "write", "execute" or a combination thereof,
- the list of ``role`` names the rule applies to


Using ``access`` rules, the permissions check algorithm can be described formally as follows ::


    ## The user U requests a permission mode P (e.g. "read") for an object O

    let currentObject = O
    let userRoles = "roles" of the user U

    loop

        if currentObject has property "access"

            ## Check explicit access rules:

            for each Rule in currentObject.access
                if (Rule.roles contains any of userRoles) and (Rule.mode contains P)
                    if Rule.type is "allow", return Access Granted
                    if Rule.type is "deny",  return Access Denied
                end if
            end for

        end if

        ## At this point, the currentObject either has no "access" rules,
        ## or none of these rules match user's roles.
        ## Check the parent object if it exists

        if currentObject has a "parent"
            let currentObject = currentObject.parent
            continue loop
        end if

        ## At this point, we've reached the root object
        ## and still haven't found any matching rule.
        ## Use the default rule, which is "Deny all requests"

        return Access Denied

    end loop


Roles
-----

There are some predefined roles that have special meaning in GBD WebSuite:

TABLE
   *guest* ~ Not logged-in user
   *user* ~ Any logged-in user
   *everyone* ~ All users, logged-in and guests
   *admin* ~ Administrator. Users that have this role are automatically granted access to all resources
/TABLE

Othewise, you can use arbitrary role names, but they must be valid identifers (i.e. start with a latin letter and only contain letters, digits and underscores).


Authorization strategies
------------------------

Since access rules are inherited, the first thing you have to configure is the root ``access`` list. If your projects are mostly public (or when you don't need any authorization at all), you can grant ``read`` and ``write`` to "everyone" ::

    ## in the main config:

    "access": [
        {
            "type": "allow",
            "mode": ["read", "write"],
            "role": ["everyone"]
        }
    ]

Now, if you need to restrict access to some object, e.g. a project, you need two access rules: one to allow a specific role, and one to deny "everyone" ::

    ## in the project config:

    "access": [
        {
            "type": "allow",
            "mode": ["read", "write"],
            "role": ["members"]
        },
        {
            "type": "deny",
            "mode": ["read", "write"],
            "role": ["everyone"]
        }
    ]

On the other side, if most of your projects require a login, it's easier to start with a "deny all" rule ::

    ## in the main config:

    "access": [
        {
            "type": "deny",
            "mode": ["read", "write"],
            "role": ["everyone"]
        }
    ]

and then explicitly allow access to specific objects ::

    # in the project config:

    "access": [
        {
            "type": "allow",
            "mode": ["read", "write"],
            "role": ["members"]
        }
    ]

Usually, there's no need to configure ``execute`` rights specifically, but if you decide to do so, pay attention that at least ``asset`` and ``auth`` actions are executable by everyone, otherwise your users wouldn't even be able to login!


Authorization providers
-----------------------

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

        "roles": [

            ## LDAP user "euler" has the GWS role "moderators":

            {
                "matches": "(&(cn=euler))",
                "role": "moderators"
            },

            ## all members of the LDAP group "mathematicians" have the GWS role "members":

            {
                "memberOf": "(&(ou=mathematicians))",
                "role": "members"
            }
        ]
    }
