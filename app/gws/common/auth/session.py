# @TODO: support different storage backends
# @TODO cache users in sessions for now, decide when to contact a provider later on

import time

import gws
import gws.config
import gws.tools.json2 as json2

import gws.types as t

from .stores import sqlite as store
from . import util

class Session:
    def __init__(self, rec, user, data):
        self.rec = rec
        self.uid = rec['uid']
        self.user = user
        self.data = data or {}
        self.changed = False

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, val):
        self.data[key] = val
        self.changed = True


class Manager:
    def _session_object(self, rec):
        if rec:
            prov: t.IAuthProvider = gws.config.root().find('gws.ext.auth.provider', rec['provider_uid'])
            if not prov:
                gws.log.error(f'auth provider not found: {rec!r}')
                return
            user = util.unserialize_user(rec['str_user'])
            data = json2.from_string(rec['str_data'])
            return Session(rec, user, data)

    def find(self, uid):
        rec = store.find(uid)
        if not rec:
            return

        td = int(time.time()) - rec['updated']
        if td > gws.config.root().var('auth.session.lifeTime'):
            gws.log.info(f'session uid={uid!r} EXPIRED time={td!r}')
            store.delete(uid)
            return

        gws.log.info(f'session uid={uid!r} active time={td!r}')
        return self._session_object(rec)

    def create_for(self, user):
        self.cleanup()
        uid = store.create(
            user.provider.uid,
            user.uid,
            util.serialize_user(user))
        return self._session_object(store.find(uid))

    def cleanup(self):
        store.cleanup(gws.config.root().var('auth.session.lifeTime'))

    def delete(self, sess):
        store.delete(sess.uid)

    def update(self, sess):
        str_data = json2.to_string(sess.data)
        store.update(sess.uid, str_data)

    def delete_all(self):
        store.drop()

    def get_all(self):
        return [{
            'user_uid': r['user_uid'],
            'created': r['created'],
            'updated': r['updated'],
        } for r in store.get_all()]

    def count(self):
        return store.count()