"""Account plugin.

This plugin manages user accounts. Accounts are stored in a database table. This plugin provides facilities for managing
and editing account data and thus is different from the "sql" authorization provider, which can only authorize users.

The accounts DB table can have an arbitrary name and should contain the following columns: ::

    id              int primary key generated always as identity,

    email           text not null,  -- user email
    status          int default 0,  -- use status

    password        text,           -- password hash
    mfauid          text,           -- MFA adapter uid, if used
    mfasecret       text,           -- MFA secret value

    tc              text,           -- storage for a temporary code
    tctime          int,            -- temporary code timestamp
    tccategory      text,           -- temporary code category


The table can also contain further columns for user info and data. These columns can be configured in the account models
and thus made editable for account administrators and/or end users.

This plugin provides the global ``account`` helper, which contains database models and various options.

Additionally, the following components are defined:

- account administration: action ``accountAdmin`` and the client component ``Sidebar.AccountAdmin``.
- account management for end users: action ``account`` and the client component ``Dialog.Account``. Also used for the onboarding procedure.
- authorization provider ``account``. Authorizes users based on the accounts table.

These components are optional and can be used together or separately. All components require the global helper to be configured.

Configuration example: ::


    @# global configuration

    helpers+ {
        type "account"
        adminModel { ... definition for the administrator model }
        options...
    }

    auth.providers+ {
        type "account"
    }

    @# some "admin" project

    projects+ {
        ....
        action {
            type "accountAdmin"
            permissions.read "allow admin, deny all"
        }
        client.addElements {
            tag "Sidebar.AccountAdmin"
        }
    }

    @# some "user" project

    projects+ {
        ....
        action {
            type "account"
            permissions.read "allow user, deny all"
        }
        client.addElements {
            tag "Dialog.Account"
        }
    }

"""
