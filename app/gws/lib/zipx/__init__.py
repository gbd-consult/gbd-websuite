"""Zipfile wrappers."""

import io
import os
import shutil
import zipfile

import gws


class Error(gws.Error):
    pass


def zip_to_path(path: str, *sources, flat: bool = False) -> int:
    """Create a zip archive in a file.

    Args:
        path: Path to the zip archive.
        sources: Paths or dicts to zip.
        flat: If ``True`` base names are being kept in zip archive,
                else whole paths are being kept in zip archive. Default is ``False``

    Returns:
        The amount of files in the zip archive.
    """

    return _zip(path, sources, flat)


def zip_to_bytes(*sources, flat: bool = False) -> bytes:
    """Create a zip archive in memory.

    Args:
        sources: Paths or dicts to zip.
        flat: If ``True`` only base names will be returned,
                else the whole paths will be returned. Default is ``False``.

    Returns:
        The names of the file paths encoded in bytes.
    """

    with io.BytesIO() as fp:
        cnt = _zip(fp, sources, flat)
        return fp.getvalue() if cnt else b''


def unzip_path(path: str, target_dir: str, flat: bool = False) -> int:
    """Unpack a zip archive into a directory.

    Args:
        path: Path to the zip archive.
        target_dir: Path to the target directory.
        flat: If ``True`` omit path and consider only base name of files in the zip archive,
                else complete paths are considered of files in the zip archive. Default is ``False``.

    Returns:
        The amount of unzipped files.
    """

    return _unzip(path, target_dir, None, flat)


def unzip_bytes(source: bytes, target_dir: str, flat: bool = False) -> int:
    """Unpack a zip archive in memory into a directory.

    Args:
        source: Path to the zip archive.
        target_dir: Path to the target directory.
        flat: If ``True`` omit path and consider only base name of files in the zip archive,
                else complete paths are considered of files in the zip archive. Default is ``False``.

    Returns:
        The amount of unzipped files.
    """

    with io.BytesIO(source) as fp:
        return _unzip(fp, target_dir, None, flat)


def unzip_path_to_dict(path: str, flat: bool = False) -> dict:
    """Unpack a zip archive into a dict.

    Args:
        path: Path to the zip archive.
        flat: If ``True`` then the dictionary contains the base names of the unzipped files,
                else it contains the whole path. Default is ``False``.

    Returns:
        A dictionary containing all the file paths or base names.
    """

    dct = {}
    _unzip(path, None, dct, flat)
    return dct


def unzip_bytes_to_dict(source: bytes, flat: bool = False) -> dict:
    """Unpack a zip archive in memory into a dict.

    Args:
        source: Path to zip archive.
        flat: If ``True`` then the dictionary contains the base names of the unzipped files,
                else it contains the whole path. Default is ``False``.

    Returns:
        A dictionary containing all the file paths or base names.
    """

    with io.BytesIO(source) as fp:
        dct = {}
        _unzip(fp, None, dct, flat)
        return dct


##

def _zip(target, sources, flat):
    files = []
    dct = {}

    for src in sources:

        if isinstance(src, dict):
            for name, content in src.items():
                dct[os.path.basename(name) if flat else name] = content
            continue

        if os.path.isdir(src):
            for p in _scan_dir(src):
                if flat:
                    files.append([p, os.path.basename(p)])
                else:
                    files.append([p, os.path.relpath(p, src)])
            continue

        if os.path.isfile(src):
            if flat:
                files.append([src, os.path.basename(src)])
            else:
                files.append([src, src])
            continue

        raise Error(f'zip: invalid argument: {src!r}')

    if not files and not dct:
        return 0

    cnt = 0

    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, arcname in files:
            zf.write(filename, arcname)
            cnt += 1
        for name, content in dct.items():
            zf.writestr(name, content)
            cnt += 1

    return cnt


def _unzip(source, target_dir, target_dict, flat):
    cnt = 0

    with zipfile.ZipFile(source, 'r') as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue

            path = zi.filename
            base = os.path.basename(path)

            if path.startswith(('/', '.')) or '..' in path or not base:
                gws.log.warning(f'unzip: invalid file name: {path!r}')
                continue

            if target_dict is not None:
                with zf.open(path) as src:
                    target_dict[base if flat else path] = src.read()
                continue

            if flat:
                dst = os.path.join(target_dir, base)
            else:
                dst = os.path.join(target_dir, *path.split('/'))
                os.makedirs(os.path.dirname(dst), exist_ok=True)

            with zf.open(path) as src, open(dst, 'wb') as fp:
                shutil.copyfileobj(src, fp)
                cnt += 1

    return cnt


def _scan_dir(source_dir):
    paths = []

    for de in os.scandir(source_dir):
        if de.is_file():
            paths.append(de.path)
        elif de.is_dir():
            paths.extend(_scan_dir(de.path))

    return paths
