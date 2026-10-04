"""Authorization provider for the accounts table."""

from typing import Optional, cast

import gws
import gws.base.auth

from . import core, helper


@gws.ext.config.authProvider('account')
class Config(gws.base.auth.provider.Config):
    """Authentication against the accounts table of the account helper."""
    pass


@gws.ext.object.authProvider('account')
class Object(gws.base.auth.provider.Object):
    """Authorization provider that authenticates users against the accounts table of the account helper."""

    h: helper.Object
    """The account helper."""

    def configure(self):
        self.h = cast(helper.Object, self.root.app.helper('account'))

    def authenticate(self, method, credentials):
        try:
            account = self.h.get_account_by_credentials(credentials, expected_status=core.Status.active)
        except helper.Error as exc:
            raise gws.ForbiddenError() from exc

        if account:
            return self._make_user(account)

    def get_user(self, local_uid):
        account = self.h.get_account_by_id(local_uid)
        if account:
            return self._make_user(account)

    def _make_user(self, account: dict) -> gws.User:
        """Create a user from an account record.

        The record is converted with ``gws.base.auth.user.from_record``, the primary key becomes the local user uid.
        """

        user_rec = {}

        for k, v in account.items():
            if k == self.h.adminModel.uidName:
                user_rec['localUid'] = str(v)
            else:
                user_rec[k] = v

        return gws.base.auth.user.from_record(self, user_rec)
