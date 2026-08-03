"""Tests for the upload helper."""

import gws
import gws.test.util as u
import gws.plugin.upload_helper as uh


@u.fixture(autouse=True)
def locks_dir():
    gws.u.ensure_dir(gws.c.LOCKS_DIR)


def _helper(max_size=1000):
    obj = uh.Object()
    obj.initialize(uh.Config(maxSize=max_size))
    return obj


def _send(obj, content=b'', uid='', number=0, count=1, size=None, file_name='file_1'):
    p = uh.ChunkRequest(
        uploadUid=uid,
        fileName=file_name,
        totalSize=len(content) if size is None else size,
        chunkNumber=number,
        chunkCount=count,
        content=content,
    )
    return obj.handle_chunk_request(None, p).uploadUid


def _send_all(obj, contents, file_name='file_1'):
    total = sum(len(c) for c in contents)
    uid = ''
    for n, c in enumerate(contents):
        uid = _send(obj, content=c, uid=uid, number=n, count=len(contents), size=total, file_name=file_name)
    return uid


def test_single_chunk():
    obj = _helper()
    uid = _send_all(obj, [b'content_1'])
    up = obj.get_upload(uid)

    assert up.uid == uid
    assert up.fileName == 'file_1'
    assert up.totalSize == 9
    assert up.chunkCount == 1
    assert gws.u.read_file_b(up.path) == b'content_1'


def test_multiple_chunks():
    obj = _helper()
    uid = _send_all(obj, [b'aaa', b'bbb', b'ccc'])
    up = obj.get_upload(uid)

    assert up.chunkCount == 3
    assert gws.u.read_file_b(up.path) == b'aaabbbccc'


def test_chunks_in_any_order():
    obj = _helper()
    uid = _send(obj, content=b'ccc', number=2, count=3, size=9)
    _send(obj, content=b'aaa', uid=uid, number=0, count=3, size=9)
    _send(obj, content=b'bbb', uid=uid, number=1, count=3, size=9)

    assert gws.u.read_file_b(obj.get_upload(uid).path) == b'aaabbbccc'


def test_uid_is_created_once():
    obj = _helper()
    uid_1 = _send(obj, content=b'aaa', number=0, count=2, size=6)
    uid_2 = _send(obj, content=b'bbb', uid=uid_1, number=1, count=2, size=6)

    assert uid_1 == uid_2


def test_get_upload_is_repeatable():
    obj = _helper()
    uid = _send_all(obj, [b'aaa', b'bbb'])
    up_1 = obj.get_upload(uid)
    up_2 = obj.get_upload(uid)

    assert up_1.path == up_2.path
    assert gws.u.read_file_b(up_2.path) == b'aaabbb'


def test_chunk_files_are_removed_after_finalize():
    obj = _helper()
    uid = _send_all(obj, [b'aaa', b'bbb'])
    obj.get_upload(uid)

    assert not gws.u.is_file(uh._base_path(uid, 0))
    assert not gws.u.is_file(uh._base_path(uid, 1))


def test_incomplete_upload():
    obj = _helper()
    uid = _send(obj, content=b'aaa', number=0, count=2, size=6)

    with u.raises(uh.Error):
        obj.get_upload(uid)


def test_size_mismatch():
    obj = _helper()
    uid = _send(obj, content=b'aaa', number=0, count=1, size=100)

    with u.raises(uh.Error):
        obj.get_upload(uid)


def test_unknown_uid():
    obj = _helper()

    with u.raises(uh.Error):
        obj.get_upload('abc123')


def test_invalid_uid():
    obj = _helper()

    with u.raises(uh.Error):
        obj.get_upload('../etc/passwd')


def test_invalid_chunk_number():
    obj = _helper()
    uid = _send(obj, content=b'aaa', number=0, count=2, size=6)

    with u.raises(gws.BadRequestError):
        _send(obj, content=b'bbb', uid=uid, number=2, count=2, size=6)

    with u.raises(gws.BadRequestError):
        _send(obj, content=b'bbb', uid=uid, number=-1, count=2, size=6)


def test_invalid_total_size():
    obj = _helper(max_size=1)

    for size in (0, -1, 1024 * 1024 + 1):
        with u.raises(gws.BadRequestError):
            _send(obj, content=b'aaa', count=1, size=size)


def test_max_total_size():
    obj = _helper(max_size=1)
    uid = _send(obj, content=b'aaa', count=1, size=1024 * 1024)

    assert uid


def test_invalid_chunk_count():
    obj = _helper(max_size=1)

    assert obj.maxChunkCount == 2

    for count in (0, -1, 3):
        with u.raises(gws.BadRequestError):
            _send(obj, content=b'aaa', number=0, count=count, size=6)


def test_max_chunk_count_is_derived_from_max_size():
    assert _helper(max_size=1).maxChunkCount == 2
    assert _helper(max_size=1000).maxChunkCount == 2048


def test_default_max_size():
    obj = uh.Object()
    obj.initialize(uh.Config())

    assert obj.maxSize == 1000 * 1024 * 1024
    assert obj.maxChunkCount == 2048
