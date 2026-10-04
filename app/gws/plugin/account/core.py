"""Constants for user accounts."""

import gws


class Category:
    """Categories of temporary codes, also used as template subject prefixes for emails."""

    onboarding = 'onboarding'
    """Code for the onboarding procedure."""
    resetPassword = 'resetPassword'
    """Code for a password reset, not used yet."""


class Status(gws.Enum):
    """Account status."""

    new = 0
    """Created or reset, onboarding not started."""
    onboarding = 1
    """Onboarding in progress."""
    active = 10
    """Active, the user can log in."""


class Columns:
    """Column names in the accounts table."""

    username = 'username'
    """Login name, not used: the login column is configured with ``usernameColumn``."""
    email = 'email'
    """Email address."""
    status = 'status'
    """Account status, see ``Status``."""
    password = 'password'
    """Password hash."""
    mfaUid = 'mfauid'
    """Uid of the selected MFA adapter, empty for no MFA."""
    mfaSecret = 'mfasecret'
    """MFA secret."""
    tc = 'tc'
    """Temporary code."""
    tcTime = 'tctime'
    """Creation time of the temporary code, as a Unix timestamp."""
    tcCategory = 'tccategory'
    """Category of the temporary code, see ``Category``."""
