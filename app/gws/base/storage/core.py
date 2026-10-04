"""Storage object."""

from typing import Optional

import gws
import gws.lib.jsonx


class Verb(gws.Enum):
    """Storage action verbs."""

    read = 'read'
    """Read an entry."""
    write = 'write'
    """Write an entry."""
    list = 'list'
    """List entries."""
    delete = 'delete'
    """Delete an entry."""


class State(gws.Data):
    """Storage state."""

    names: list[str]
    """List of entry names."""
    canRead: bool
    """User can read entries."""
    canWrite: bool
    """User can write entries."""
    canCreate: bool
    """User can create entries."""
    canDelete: bool
    """User can delete entries."""


class Request(gws.Request):
    """Storage request."""

    verb: Verb
    """Action to perform."""
    entryName: Optional[str]
    """Entry name, required for read, write and delete."""
    entryData: Optional[dict]
    """Entry data to write."""


class Response(gws.Response):
    """Storage response."""

    data: Optional[dict]
    """Entry data for a read request."""
    state: State
    """Storage state after the request."""


class Config(gws.ConfigWithAccess):
    """Storage for entries saved by users."""

    providerUid: Optional[str]
    """UID of the storage provider."""
    categoryName: Optional[str]
    """Category under which entries are stored in the provider."""


class Props(gws.Props):
    state: State


class Object(gws.Node):
    """Storage object, gives users access to entries of one category in a storage provider."""

    storageProvider: gws.StorageProvider
    """Storage provider."""
    categoryName: str
    """Category under which entries are stored."""

    def configure(self):
        self.configure_provider()
        self.categoryName = self.cfg('categoryName')

    def configure_provider(self):
        """Find the storage provider by ``providerUid``, or use the first provider.

        Returns:
            ``True``.

        Raises:
            ``gws.Error``: If no provider is found.
        """
        self.storageProvider = self.root.app.storageMgr.find_provider(self.cfg('providerUid'))
        if not self.storageProvider:
            raise gws.Error(f'storage provider not found')
        return True

    def props(self, user):
        return gws.Props(
            state=self.get_state_for(user),
        )

    def get_state_for(self, user):
        """Return the storage state for a user.

        Args:
            user: The user.

        Returns:
            The entry names, if the user can read them, and the user's permissions.
        """
        return State(
            names=self.storageProvider.list_names(self.categoryName) if user.can_read(self) else [],
            canRead=user.can_read(self),
            canWrite=user.can_write(self),
            canDelete=user.can_delete(self),
            canCreate=user.can_create(self),
        )

    def handle_request(self, req: gws.WebRequester, p: Request) -> Response:
        """Handle a storage request from the client.

        Writing an existing entry requires the ``write`` permission, writing a new
        one requires ``create``. Entry data is stored as JSON.

        Args:
            req: Web requester.
            p: Storage request.

        Returns:
            The entry data for a read request, and the new storage state.

        Raises:
            ``gws.ForbiddenError``: If the user lacks the permission or the entry name or data is missing.
        """
        state = self.get_state_for(req.user)
        data = None

        if p.verb == Verb.list:
            if not state.canRead:
                raise gws.ForbiddenError()
            pass

        if p.verb == Verb.read:
            if not state.canRead or not p.entryName:
                raise gws.ForbiddenError()
            rec = self.storageProvider.read(self.categoryName, p.entryName)
            if rec:
                data = gws.lib.jsonx.from_string(rec.data)

        if p.verb == Verb.write:
            if not p.entryName or not p.entryData:
                raise gws.ForbiddenError()
            if p.entryName in state.names and not state.canWrite:
                raise gws.ForbiddenError()
            if p.entryName not in state.names and not state.canCreate:
                raise gws.ForbiddenError()

            d = gws.lib.jsonx.to_string(p.entryData)
            self.storageProvider.write(self.categoryName, p.entryName, d, req.user.uid)

        if p.verb == Verb.delete:
            if not state.canDelete or not p.entryName:
                raise gws.ForbiddenError()
            self.storageProvider.delete(self.categoryName, p.entryName)

        return Response(data=data, state=self.get_state_for(req.user))
