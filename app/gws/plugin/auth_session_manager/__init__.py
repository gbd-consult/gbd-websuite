"""Session managers.

A session manager stores authentication sessions: it creates, finds, updates
and deletes them and removes expired ones. It is configured in
``auth.session``; the default type is ``sqlite``. Session life times
(``lifeTime``, ``maxLifeTime``) are handled by the base class
``gws.base.auth.session_manager``.

Subpackages
-----------

- ``sqlite`` - sessions stored in an SQLite database file.

Example::

    auth.session {
        type "sqlite"
        lifeTime "1h"
    }
"""
