"""Token authorization method for QField clients."""

import datetime
import re

import gws
import gws.base.auth
import gws.lib.datetimex as dtx


@gws.ext.config.authMethod('qfieldcloud')
class Config(gws.base.auth.method.Config):
    """Token authentication for QField clients."""


class Result(gws.Data):
    """Result of a successful authentication."""

    sess: gws.AuthSession
    """Session of the user."""
    user: gws.User
    """Authenticated user."""
    token: str
    """Session token, sent by QField in the ``Authorization`` header."""
    expiresAt: datetime.datetime
    """Expiry time of the session."""


@gws.ext.object.authMethod('qfieldcloud')
class Object(gws.base.auth.method.Object):
    """QField Cloud authorization method.

    Token authorization for QField clients. The ``qfieldcloud`` action creates
    sessions with this method on login and accepts only tokens of its sessions.
    """

    def configure(self):
        self.uid = 'gws.plugin.qfieldcloud.auth'

    def authenticate_from_credentials(self, credentials: gws.Data, req: gws.WebRequester) -> Result:
        """Authenticate a user by username and password and create a session.

        Args:
            credentials: Data with ``username`` and ``password``.
            req: Web request.

        Returns:
            Authentication result with the new session and its expiry time.

        Raises:
            ``gws.AuthenticationError``: If the credentials are invalid.
        """
        am = self.root.app.authMgr
        user = am.authenticate(self, credentials, req)
        if not user:
            raise gws.AuthenticationError('invalid username or password')

        sess = am.sessionMgr.create(self, user)
        return Result(
            sess=sess,
            user=user,
            token=sess.uid,
            expiresAt=dtx.add(seconds=am.sessionMgr.lifeTime),
        )

    def authenticate_from_token(self, req: gws.WebRequester) -> Result:
        """Authenticate a request by the ``Authorization: Token ...`` header and touch the session.

        Args:
            req: Web request.

        Returns:
            Authentication result with the existing session and its new expiry time.

        Raises:
            ``gws.AuthenticationError``: If the header is missing, or the token is invalid or belongs to another method.
            ``gws.ForbiddenError``: If the method cannot be used in this context, e.g. without a secure connection.
        """
        h = req.header('Authorization', '')
        m = re.match(r'^Token (.+)$', h)
        if not m:
            raise gws.AuthenticationError('qfieldcloud_auth: missing or invalid Authorization header')

        token = m.group(1)
        am = self.root.app.authMgr
        if not am.can_use_method(req, self):
            raise gws.ForbiddenError('qfieldcloud_auth: insecure_context')

        sess = am.sessionMgr.get(token)
        if not sess:
            raise gws.AuthenticationError('qfieldcloud_auth: invalid or expired token')
        if not sess.method or sess.method.uid != self.uid:
            raise gws.AuthenticationError(f'qfieldcloud_auth: wrong method {sess.method=}')

        am.sessionMgr.touch(sess)
        gws.log.debug(f'qfieldcloud_auth: ok: {token=} {sess.user.uid=} {sess.user.loginName=}')

        return Result(
            sess=sess,
            user=sess.user,
            token=sess.uid,
            expiresAt=dtx.add(seconds=am.sessionMgr.lifeTime),
        )
