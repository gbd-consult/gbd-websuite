"""User accounts stored in a database table.

This plugin manages user accounts in a PostgreSQL table. Unlike the ``sql`` authorization provider, which can only
authorize users, it also provides account administration and a self-service onboarding procedure for new users.
Users do not register themselves: an administrator creates an account, and the user activates it via a link
sent by email.

Accounts table
--------------

The table can have an arbitrary name and should contain the following columns::

    id              int primary key generated always as identity,

    email           text not null,  -- user email
    status          int default 0,  -- account status

    password        text,           -- password hash
    mfauid          text,           -- MFA adapter uid, if used
    mfasecret       text,           -- MFA secret value

    tc              text,           -- storage for a temporary code
    tctime          int,            -- temporary code timestamp
    tccategory      text,           -- temporary code category

The table can also contain further columns for user info and data. These columns can be configured in the account
models and thus made editable for account administrators. The login name is read from the ``email`` column,
or from another column given by ``usernameColumn``. Passwords are hashed and checked in SQL, by default with
pgcrypto's ``crypt``.

Account status
--------------

An account is ``new`` (0) after it was created or reset, ``onboarding`` (1) while the user is activating it,
and ``active`` (10) afterwards. Only active accounts can log in.

Onboarding
----------

When an administrator creates or resets an account, its password and MFA secret are cleared, the status is set to
``new``, and, if ``onboardingUrl`` is configured, an email with the link ``<onboardingUrl>?onboarding=<code>`` is sent
to the account's email address. The temporary code expires after ``tcLifeTime`` and can be used once; every step
of the procedure issues a new one. On the onboarding page, the user confirms the email address and sets a password,
then selects one of the configured MFA methods, if any. The account becomes active and the client is redirected
to ``onboardingCompletionUrl``.

Emails are rendered from the helper's templates with the subjects ``onboarding.emailSubject`` and
``onboarding.emailBody`` and sent by the ``email`` helper. The templates receive the ``account`` record
and the ``url``.

Modules
-------

``helper``
    The global ``account`` helper: configuration, account queries and updates, temporary codes, passwords, MFA,
    emails. Also implements the edit API for the administration model. All other components use it.

``admin_action``
    Action ``accountadmin``: administration of accounts in the client (``Sidebar.AccountAdmin``),
    including the reset of an account.

``account_action``
    Action ``account``: the onboarding procedure for end users (``Account.Dialog``).

``auth_provider``
    Authorization provider ``account``: authenticates active users against the accounts table.

``cli``
    The CLI command ``accountReset``, which resets accounts by their ids.

``core``
    Constants: column names, account status values, temporary code categories.

The components are optional and can be used together or separately. All of them require the helper to be configured.

Example::

    @# global configuration

    helpers+ {
        type "account"
        usernameColumn "login"
        onboardingUrl "https://example.com/project/user_account"
        mfa [
            { mfaUid ""              title "No multi-factor authentication" }
            { mfaUid "AUTH_MFA_TOTP" title "Authenticator app" }
        ]
        adminModel {
            type "postgres"
            tableName "edit.accounts"
            isEditable true
            permissions.edit "allow admin, deny all"
            ...
        }
        templates+ { subject "onboarding.emailSubject" type "text" text "Activate your account" }
        templates+ { subject "onboarding.emailBody" type "text" text "Activate your account: {{url}}" }
    }

    auth.providers+ {
        type "account"
    }

    @# administration project

    projects+ {
        ...
        actions+ {
            type "accountadmin"
            permissions.read "allow admin, deny all"
        }
        client.addElements+ { tag "Sidebar.AccountAdmin" }
    }

    @# onboarding project, the target of onboardingUrl

    projects+ {
        uid "user_account"
        ...
        actions+ {
            type "account"
            permissions.read "allow all"
        }
        client.addElements+ { tag "Account.Dialog" }
    }
"""
