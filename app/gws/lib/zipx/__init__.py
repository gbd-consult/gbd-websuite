"""Zipfile wrappers."""

import io
import os
import shutil
import zipfile

import gws


class Error(gws.Error):
    pass


def zip_to_path(path: str, sources: list[str | dict], base_dir: str = '', flat: bool = False) -> int:
    """Create a zip archive in a file.

    Args:
        path: Path to the archive.
        sources: A list of paths or dicts to zip. If a dict is given, 
                its keys are file names in the archive and its values are the file contents.
        base_dir: If given, this path is stripped from the beginning of the file paths in the archive.
        flat: If ``True`` only base names are being kept in archive.

    Returns:
        The amount of files in the archive.
    """

    return _zip(path, sources, base_dir, flat)


def zip_to_bytes(sources: list[str | dict], base_dir: str = '', flat: bool = False) -> bytes:
    """Create a zip archive in memory.

    Args:
        sources: A list of paths or dicts to zip. If a dict is given, 
                its keys are file names in the archive and its values are the file contents.
        base_dir: If given, this path is stripped from the beginning of the file paths in the archive.
        flat: If ``True`` only base names are being kept in archive.

    Returns:
        The zipped content as bytes.
    """

    with io.BytesIO() as fp:
        cnt = _zip(fp, sources, base_dir, flat)
        return fp.getvalue() if cnt else b''


def unzip_path(path: str, target_dir: str, flat: bool = False) -> int:
    """Unpack a zip archive into a directory.

    Args:
        path: Path to the zip archive.
        target_dir: Path to the target directory.
        flat: If ``True`` omit path and consider only base name of files in the zip archive,
                else complete paths are considered of files in the zip archive. Default is ``False``.

    Returns:
        The number of unzipped files.
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
        The number of unzipped files.
    """

    with io.BytesIO(source) as fp:
        return _unzip(fp, target_dir, None, flat)


def unzip_path_to_dict(path: str, flat: bool = False) -> dict[str, bytes]:
    """Unpack a zip archive into a dict.

    Args:
        path: Path to the zip archive.
        flat: If ``True`` then the result contains the base names of the unzipped files,
                else it contains the whole path. Default is ``False``.

    Returns:
        A dictionary whose keys are the file paths or base names and values are the file contents.
    """

    dct = {}
    _unzip(path, None, dct, flat)
    return dct


def unzip_bytes_to_dict(source: bytes, flat: bool = False) -> dict[str, bytes]:
    """Unpack a zip archive in memory into a dict.

    Args:
        source: Path to zip archive.
        flat: If ``True`` then the result contains the base names of the unzipped files,
                else it contains the whole path. Default is ``False``.

    Returns:
        A dictionary whose keys are the file paths or base names and values are the file contents.
    """

    with io.BytesIO(source) as fp:
        dct = {}
        _unzip(fp, None, dct, flat)
        return dct


##


def _zip(target, sources, base_dir, flat):
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
