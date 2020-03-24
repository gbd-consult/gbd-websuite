"""Manage chunked uploads."""

import os
import shutil

import gws
import gws.tools.json2
import gws.tools.os2
import gws.web.error

import gws.types as t


class Error(gws.Error):
    pass


class UploadRecord(t.Data):
    uid: str
    name: str
    path: str


class UploadChunkParams(t.Params):
    uid: str
    name: str
    totalSize: int
    content: bytes
    chunkNumber: int
    chunkCount: int


class UploadChunkResponse(t.Response):
    uid: str


_UPLOAD_DIR = gws.TMP_DIR + '/uploads'


def upload_chunk(p: UploadChunkParams) -> UploadChunkResponse:
    try:
        uid = save_chunk(
            uid=p.uid,
            name=p.name,
            content=p.content,
            total_size=p.totalSize,
            chunk_number=p.chunkNumber,
            chunk_count=p.chunkCount
        )
        return UploadChunkResponse(uid=uid)
    except Error as e:
        gws.log.error(e)
        raise gws.web.error.BadRequest()


def save_chunk(uid: str, name: str, content: bytes, total_size: int, chunk_number: int, chunk_count: int) -> str:
    dir = gws.ensure_dir(_UPLOAD_DIR)

    if chunk_number == 1:
        uid = gws.random_string(64)
        status = t.Data(
            name=name,
            total_size=total_size,
            chunk_count=chunk_count,
        )
        gws.tools.json2.to_path(f'{dir}/{uid}.json', status)
    elif not uid.isalnum():
        raise Error(f'upload {uid!r}: invalid uid')
    else:
        try:
            status = t.Data(gws.tools.json2.from_path(f'{dir}/{uid}.json'))
        except gws.tools.json2.Error:
            status = None

    if not status:
        raise Error(f'upload {uid!r}: invalid status')

    if chunk_number < 1 or chunk_number > status.chunk_count:
        raise Error(f'upload {uid!r}: invalid chunk number')

    gws.write_file(f'{dir}/{uid}.{chunk_number}', content, 'wb')
    return uid


def get(uid: str) -> UploadRecord:
    dir = gws.ensure_dir(_UPLOAD_DIR)

    try:
        status = t.Data(gws.tools.json2.from_path(f'{dir}/{uid}.json'))
    except gws.tools.json2.Error:
        status = None

    if not status:
        raise Error(f'upload {uid!r}: not found')

    path = f'{dir}/{uid}.all'

    if os.path.isfile(path):
        return UploadRecord(uid=uid, path=path, name=status.name)

    # @TODO this should use a system lock

    chunks = [f'{dir}/{uid}.{n}' for n in range(1, status.chunk_count + 1)]

    if not all(os.path.isfile(c) for c in chunks):
        raise Error(f'upload {uid!r}: incomplete')

    tmp_path = path + '.' + gws.random_string(6)

    with open(tmp_path, 'wb') as fp_all:
        for c in chunks:
            try:
                with open(c, 'rb') as fp:
                    shutil.copyfileobj(fp, fp_all)
            except (OSError, IOError) as e:
                raise Error(f'upload {uid!r}: read error') from e

    if gws.tools.os2.file_size(tmp_path) != status.total_size:
        raise Error(f'upload {uid!r}: invalid file size')

    try:
        os.rename(tmp_path, path)
    except OSError:
        raise Error(f'upload {uid!r}: move error') from e

    for c in chunks:
        gws.tools.os2.unlink(c)

    return UploadRecord(uid=uid, path=path, name=status.name)


def delete(uid: str):
    dir = gws.ensure_dir(_UPLOAD_DIR)

    for p in gws.tools.os2.find_files(dir):
        if p.startswith(uid + '.'):
            gws.tools.os2.unlink(p)
