from typing import Optional, cast

import gws
import gws.config
import gws.lib.datetimex as dtx
import gws.lib.cli as cli

from . import manager

gws.ext.new.cli('auth')


class RemoveParams(gws.CliParams):
    older: Optional[int]
    """Remove sessions older than N seconds."""
    uid: Optional[list[str]]
    """Remove specific sessions."""
    all: Optional[bool]
    """Remove all sessions."""


class Object(gws.Node):

    @gws.ext.command.cli('authSessions')
    def sessions(self, p: gws.EmptyRequest):
        """Show active authorization sessions"""

        root = gws.config.load()
        sm = root.app.authMgr.sessionMgr

        sm.cleanup()

        rows = []

        for s in sm.get_all():
            dc = dtx.total_difference(s.created).seconds
            du = dtx.total_difference(s.updated).seconds
            rows.append((du, dict(
                uid=s.uid,
                user=s.user.loginName,
                auth=s.method.extType,
                started=dtx.to_iso_string(s.created, sep=' ') + f' ({dc} sec)',
                updated=dtx.to_iso_string(s.updated, sep=' ') + f' ({du} sec)',
            )))

        # oldest first
        rows.sort(key=lambda r: -r[0])

        cli.info('')
        cli.info(f'Active sessions: {len(rows)}')
        cli.info(cli.text_table([r[1] for r in rows], header='auto'))

    @gws.ext.command.cli('authSessrem')
    def sessrem(self, p: RemoveParams):
        """Remove authorization sessions."""

        root = gws.config.load()
        sm = root.app.authMgr.sessionMgr

        n = 0

        for s in sm.get_all():
            du = dtx.total_difference(s.updated).seconds
            if p.all:
                sm.delete(s)
                n += 1
                continue
            if p.older and du > p.older:
                sm.delete(s)
                n += 1
                continue
            if p.uid and s.uid in p.uid:
                sm.delete(s)
                n += 1
                continue

        cli.info('')
        cli.info(f'Removed sessions: {n}')
