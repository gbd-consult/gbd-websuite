"""The ``annotate`` action."""

from typing import Optional

import gws
import gws.base.action
import gws.base.web
import gws.base.storage


@gws.ext.config.action('annotate')
class Config(gws.base.action.Config):
    """Map annotations with measurement labels."""

    storage: Optional[gws.base.storage.Config]
    """Storage for saved annotations."""
    labels: Optional[dict]
    """Label templates per shape type."""


@gws.ext.props.action('annotate')
class Props(gws.base.action.Props):
    storage: gws.base.storage.Props
    labels: dict


@gws.ext.object.action('annotate')
class Object(gws.base.action.Object):
    """Annotate action."""

    storage: Optional[gws.base.storage.Object]
    """Storage for saved annotations, or ``None`` if not configured."""
    labels: Optional[dict]
    """Label templates per shape type, passed to the client."""

    def configure(self):
        self.storage = self.create_child_if_configured(
            gws.base.storage.Object,
            self.cfg('storage'),
            categoryName='Annotate',
        )
        self.labels = self.cfg('labels')

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            storage=self.storage,
            labels=self.labels,
        )

    @gws.ext.command.api('annotateStorage')
    def handle_storage(self, req: gws.WebRequester, p: gws.base.storage.Request) -> gws.base.storage.Response:
        """Read or write saved annotations."""
        if not self.storage:
            raise gws.NotFoundError('no storage configured')
        return self.storage.handle_request(req, p)
