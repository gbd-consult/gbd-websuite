"""Zipfile wrappers."""

import shutil
import zipfile
import os
import io

import gws.types as t


def zip_paths(zip_path: str, paths: t.List[str], flat=True) -> int:
    """Create a zip archive and add paths to it."""

    # @TODO flat=False

    cnt = 0

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            fname = os.path.basename(path)
            zf.write(path, fname)
            cnt += 1

    return cnt


def zip_dir(zip_path: str, source_dir: str, flat=True) -> int:
    """Zip a directory."""

    # @TODO flat=False

    cnt = 0

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for de in os.scandir(source_dir):
            if de.is_file():
                zf.write(de.path, de.name)
                cnt += 1

    return cnt


def unzip(zip_path: str, target_dir: str, flat=True) -> int:
    """Unzip an archive into a directory."""

    return _unzip(zip_path, target_dir, None, flat)


def unzip_to_dict(zip_path: str, flat=True) -> dict:
    """Unzip an archive into a dict."""

    d = {}
    _unzip(zip_path, None, d, flat)
    return d


def unzip_bytes(b: bytes, target_dir: str, flat=True) -> int:
    with io.BytesIO(b) as bio:
        return _unzip(bio, target_dir, None, flat)


def unzip_bytes_to_dict(b: bytes, flat=True) -> dict:
    with io.BytesIO(b) as bio:
        d = {}
        _unzip(bio, None, d, flat)
        return d


def _unzip(arg, target_dir, target_dict, flat):
    # @TODO flat=False

    cnt = 0

    with zipfile.ZipFile(arg) as zf:
        for inf in zf.infolist():
            if inf.is_dir():
                continue
            fname = os.path.basename(inf.filename)
            if not fname or fname.startswith('.'):
                continue
            with zf.open(inf.filename) as src:
                if target_dir:
                    with open(target_dir + '/' + fname, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    target_dict[fname] = src.read()
                cnt += 1

    return cnt
