"""Zipfile wrappers."""

import shutil
import zipfile
import os

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

    # @TODO flat=False

    cnt = 0

    zf = zipfile.ZipFile(zip_path)
    for inf in zf.infolist():
        if inf.is_dir():
            continue
        fname = os.path.basename(inf.filename)
        if not fname or fname.startswith('.'):
            continue
        with zf.open(inf.filename) as src, open(target_dir + '/' + fname, 'wb') as dst:
            shutil.copyfileobj(src, dst)
            cnt += 1

    return cnt
