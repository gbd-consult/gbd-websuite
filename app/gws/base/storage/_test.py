"""Tests for the storage object."""

import pytest

import gws
import gws.base.storage.core as core


class _Provider:
    def __init__(self, names):
        self.entries = {n: '{}' for n in names}

    def list_names(self, category):
        return sorted(self.entries)

    def write(self, category, name, data, user_uid):
        self.entries[name] = data


class _User:
    def __init__(self, read=False, write=False, create=False, delete=False):
        self.uid = 'user_1'
        self.perms = dict(read=read, write=write, create=create, delete=delete)

    def can_read(self, obj):
        return self.perms['read']

    def can_write(self, obj):
        return self.perms['write']

    def can_create(self, obj):
        return self.perms['create']

    def can_delete(self, obj):
        return self.perms['delete']


def _storage(names):
    obj = core.Object.__new__(core.Object)
    obj.storageProvider = _Provider(names)
    obj.categoryName = 'category_1'
    return obj


def _write(obj, user, name):
    req = gws.Data(user=user)
    p = core.Request(verb=core.Verb.write, entryName=name, entryData={'a': 1})
    return obj.handle_request(req, p)


def test_create_without_read_permission():
    obj = _storage([])
    _write(obj, _User(create=True), 'entry_1')
    assert 'entry_1' in obj.storageProvider.entries


def test_overwrite_without_read_and_write_permission_is_forbidden():
    obj = _storage(['entry_1'])
    with pytest.raises(gws.ForbiddenError):
        _write(obj, _User(create=True), 'entry_1')
    assert obj.storageProvider.entries['entry_1'] == '{}'


def test_overwrite_with_write_permission():
    obj = _storage(['entry_1'])
    _write(obj, _User(write=True), 'entry_1')
    assert obj.storageProvider.entries['entry_1'] != '{}'


def test_create_without_create_permission_is_forbidden():
    obj = _storage([])
    with pytest.raises(gws.ForbiddenError):
        _write(obj, _User(read=True, write=True), 'entry_1')
