"""Zipfile wrappers."""

import io
import os
import shutil
import zipfile

import gws


class Error(gws.Error):
    pass


def zip(path: str, *sources, flat=False) -> int:
    """Create a zip archive in a file."""

    return _zip(path, sources, flat)


def zip_to_bytes(*sources, flat=False) -> bytes:
    """Create a zip archive in memory."""

    with io.BytesIO() as fp:
        cnt = _zip(fp, sources, flat)
        return fp.getvalue() if cnt else b''


def unzip(path: str, target_dir: str, flat=False) -> int:
    """Unpack a zip archive into a directory."""

    return _unzip(path, target_dir, None, flat)


def unzip_bytes(source: bytes, target_dir: str, flat=False) -> int:
    """Unpack a zip archive in memory into a directory."""

    with io.BytesIO(source) as fp:
        return _unzip(fp, target_dir, None, flat)


def unzip_to_dict(path: str, flat=False) -> dict:
    """Unpack a zip archive into a dict."""

    dct = {}
    _unzip(path, None, dct, flat)
    return dct


def unzip_bytes_to_dict(source: bytes, flat=False) -> dict:
    """Unpack a zip archive in memory into a dict."""

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
    with zipfile.ZipFile(source, 'r') as zf:
        for inf in zf.infolist():
            fname = inf.filename
            base = os.path.basename(fname)

            if fname.startswith(('/', '.')) or '..' in fname or not base:
                raise Error(f'unzip: invalid file name: {fname!r}')

            if target_dict is not None:
                with zf.open(fname) as src:
                    target_dict[base if flat else fname] = src.read()
                continue

            if flat:
                dst = os.path.join(target_dir, base)
            else:
                dst = os.path.join(target_dir, *fname.split('/'))
                os.makedirs(os.path.dirname(dst), exist_ok=True)

            with zf.open(fname) as src, open(dst, 'wb') as fp:
                shutil.copyfileobj(src, fp)


def _scan_dir(source_dir):
    paths = []

    for de in os.scandir(source_dir):
        if de.is_file():
            paths.append(de.path)
        elif de.is_dir():
            paths.extend(_scan_dir(de.path))

    return paths
