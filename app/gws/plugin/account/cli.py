from typing import Optional, cast

import gws
import gws.config

from . import helper

class AccountResetParams(gws.CliParams):
    uid: Optional[list[str]]
    """List of account IDs to reset."""




class Object(gws.Node):
    @gws.ext.command.cli('accountReset')
    def account_reset(self, p: AccountResetParams):
        """Reset an account or multiple accounts."""

        root = gws.config.load()
        h = cast(helper.Object, root.app.helper('account'))

        for uid in p.uid:
            account = h.get_account_by_id(uid)
            if not account:
                continue
            h.reset(account)



    ##

