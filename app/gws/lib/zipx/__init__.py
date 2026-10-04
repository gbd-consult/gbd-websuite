"""Zip archive utilities.

Thin wrappers around ``zipfile`` that create and unpack zip archives in one call.

Archives are created from a list of sources (``zip_to_path``, ``zip_to_bytes``). A source is a file path,
a directory path, which is scanned recursively, or a dict of archive names and contents. Entries are compressed
with ``ZIP_DEFLATED``. Archive names are the normalized source paths, optionally with ``base_dir`` stripped,
or only the base names with ``flat=True``.

Archives are unpacked from a file or from bytes, into a directory (``unzip_path``, ``unzip_bytes``)
or into a dict of names and contents (``unzip_path_to_dict``, ``unzip_bytes_to_dict``). Directory entries
are skipped. Entries with unsafe names (absolute, starting with a dot or containing ``..``) are skipped
with a warning. With ``flat=True``, entries are unpacked by their base names, so entries with the same
base name overwrite each other.

Example::

    import gws.lib.zipx as zipx

    zipx.zip_to_path('/tmp/out.zip', ['/data/report', {'readme.txt': 'hello'}], base_dir='/data/')
    content = zipx.zip_to_bytes(['/data/a.txt', '/data/b.txt'], flat=True)

    zipx.unzip_path('/tmp/out.zip', '/tmp/unpacked')
    files = zipx.unzip_bytes_to_dict(content)  # {'a.txt': b'...', 'b.txt': b'...'}
"""

import io
import os
import shutil
import zipfile

import gws


class Error(gws.Error):
    """Zip archive error."""

    pass


def zip_to_path(path: str, sources: list[str | dict], base_dir: str = '', flat: bool = False) -> int:
    """Create a zip archive in a file.

    If there are no files to add, no archive is created.

    Args:
        path: Path to the archive.
        sources: File paths, directory paths (scanned recursively) or dicts that map
            archive names to contents (``str`` or ``bytes``).
        base_dir: Prefix to remove from the beginning of the file paths in the archive.
        flat: If ``True``, only the base names of the files are kept in the archive.

    Returns:
        The number of files in the archive.

    Raises:
        Error: If a source is neither a dict, a file nor a directory.
    """

    return _zip(path, sources, base_dir, flat)


def zip_to_bytes(sources: list[str | dict], base_dir: str = '', flat: bool = False) -> bytes:
    """Create a zip archive in memory.

    Args:
        sources: File paths, directory paths (scanned recursively) or dicts that map
            archive names to contents (``str`` or ``bytes``).
        base_dir: Prefix to remove from the beginning of the file paths in the archive.
        flat: If ``True``, only the base names of the files are kept in the archive.

    Returns:
        The archive content, or empty bytes if there are no files to add.

    Raises:
        Error: If a source is neither a dict, a file nor a directory.
    """

    with io.BytesIO() as fp:
        cnt = _zip(fp, sources, base_dir, flat)
        return fp.getvalue() if cnt else b''


def unzip_path(path: str, target_dir: str, flat: bool = False) -> int:
    """Unpack a zip archive file into a directory.

    Args:
        path: Path to the archive.
        target_dir: Path to the target directory.
        flat: If ``True``, files are unpacked by their base names directly into ``target_dir``,
            which must exist; otherwise the directories of the archive are created as needed.

    Returns:
        The number of unpacked files.
    """

    return _unzip(path, target_dir, None, flat)


def unzip_bytes(source: bytes, target_dir: str, flat: bool = False) -> int:
    """Unpack a zip archive from bytes into a directory.

    Args:
        source: The archive content.
        target_dir: Path to the target directory.
        flat: If ``True``, files are unpacked by their base names directly into ``target_dir``,
            which must exist; otherwise the directories of the archive are created as needed.

    Returns:
        The number of unpacked files.
    """

    with io.BytesIO(source) as fp:
        return _unzip(fp, target_dir, None, flat)


def unzip_path_to_dict(path: str, flat: bool = False) -> dict[str, bytes]:
    """Unpack a zip archive file into a dict.

    Args:
        path: Path to the archive.
        flat: If ``True``, the keys are the base names of the files, otherwise their paths in the archive.

    Returns:
        A dict of file names and contents.
    """

    dct = {}
    _unzip(path, None, dct, flat)
    return dct


def unzip_bytes_to_dict(source: bytes, flat: bool = False) -> dict[str, bytes]:
    """Unpack a zip archive from bytes into a dict.

    Args:
        source: The archive content.
        flat: If ``True``, the keys are the base names of the files, otherwise their paths in the archive.

    Returns:
        A dict of file names and contents.
    """

    with io.BytesIO(source) as fp:
        dct = {}
        _unzip(fp, None, dct, flat)
        return dct


##


def _zip(target, sources, base_dir, flat):
    """Write sources to a zip file or file object, return the number of files."""

    def norm_path(p):
        p = os.path.normpath(p)
        if flat:
            return os.path.basename(p)
        if base_dir:
            if p.startswith(base_dir):
                return p[len(base_dir):]
        return p

    def scan_dir(d):
        for de in os.scandir(d):
            if de.is_file():
                yield de.path
            elif de.is_dir():
                yield from scan_dir(de.path)

    args = []

    for src in sources:
        if isinstance(src, dict):
            for name, data in src.items():
                args.append((norm_path(name), None, data))
        elif os.path.isdir(src):
            for p in scan_dir(src):
                args.append((norm_path(p), p, None))       
        elif os.path.isfile(src):
            args.append((norm_path(src), src, None))
        else:
            raise Error(f'zip: invalid argument: {src!r}')

    if not args:
        return 0

    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path, data in args:
            if path:
                zf.write(path, arcname)
            else:
                zf.writestr(arcname, data)

    return len(args)


def _unzip(source, target_dir, target_dict, flat):
    """Unpack a zip file or file object into a directory or a dict, return the number of files."""

    cnt = 0

    with zipfile.ZipFile(source, 'r') as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue

            path = zi.filename.replace('\\', '/')
            base = os.path.basename(path)

            if path.startswith(('/', '.')) or '..' in path or not base:
                gws.log.warning(f'unzip: invalid file name: {path!r}')
                continue

            cnt += 1

            if target_dir:
                if flat:
                    dst = os.path.join(target_dir, base)
                else:
                    dst = os.path.join(target_dir, *path.split('/'))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)

                with zf.open(zi) as src, open(dst, 'wb') as fp:
                    shutil.copyfileobj(src, fp)
            elif target_dict is not None:
                key = base if flat else path
                with zf.open(zi) as src:
                    target_dict[key] = src.read()
            else:
                raise Error('invalid target for unzip')

    return cnt
