import gws.tools.upload
import _test.util as u

def test_chunks():
    buf = '0123456789abcdefgh'
    uid = ''
    name = 'foo'

    for n, b in enumerate(buf):
        r = u.cmd('uploadSendChunk', {
            'uid': uid,
            'name': name,
            'totalSize': len(buf),
            'content': bytes(b, 'ascii'),
            'chunkNumber': n + 1,
            'chunkCount': len(buf),
        }, binary=True).json()

        uid = r['uid']

    a = gws.tools.upload.get(uid)

    assert a.uid == uid
    assert a.name == name
    assert u.read(a.path) == buf
