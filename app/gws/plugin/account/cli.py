"""CLI commands for user accounts."""

from typing import Optional, cast

import gws

from . import helper

class AccountResetParams(gws.CliParams):
    """Parameters of the ``accountReset`` command."""

    uid: Optional[list[str]]
    """List of account IDs to reset."""




class Object(gws.Node):
    """CLI commands for user accounts."""

    @gws.ext.command.cli('accountReset')
    def account_reset(self, p: AccountResetParams):
        """Reset an account or multiple accounts."""

        root = gws.load_root()
        h = cast(helper.Object, root.app.helper('account'))

        for uid in p.uid:
            account = h.get_account_by_id(uid)
            if not account:
                continue
            h.reset(account)



    ##

