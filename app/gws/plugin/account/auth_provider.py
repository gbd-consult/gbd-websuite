"""Account-based authorization provider."""

from typing import Optional, cast

import gws
import gws.base.auth

from . import core, helper

gws.ext.new.authProvider('account')


class Config(gws.base.auth.provider.Config):
    """Account-based authorization provider."""
    pass


class Object(gws.base.auth.provider.Object):
    h: helper.Object

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
        user_rec = {}

        for k, v in account.items():
            if k == self.h.adminModel.uidName:
                user_rec['localUid'] = str(v)
            else:
                user_rec[k] = v

        return gws.base.auth.user.from_record(self, user_rec)
