"""Tests for the zipx module."""
import io
import zipfile

import gws.lib.zipx as zipx


# --- zip_to_path ---

def test_zip_to_path_single_file(tmp_path):
    f = tmp_path / 'a.txt'
    f.write_text('hello')
    z = tmp_path / 'out.zip'

    ret = zipx.zip_to_path(str(z), [str(f)])

    assert ret == 1
    assert zipfile.is_zipfile(str(z))


def test_zip_to_path_multiple_files(tmp_path):
    (tmp_path / 'a.txt').write_text('aaa')
    (tmp_path / 'b.txt').write_text('bbb')
    z = tmp_path / 'out.zip'

    ret = zipx.zip_to_path(str(z), [str(tmp_path / 'a.txt'), str(tmp_path / 'b.txt')])

    assert ret == 2


def test_zip_to_path_empty_sources(tmp_path):
    z = tmp_path / 'out.zip'

    ret = zipx.zip_to_path(str(z), [])

    assert ret == 0
    assert not z.exists()


def test_zip_to_path_directory(tmp_path):
    d = tmp_path / 'subdir'
    d.mkdir()
    (d / 'x.txt').write_text('x')
    (d / 'y.txt').write_text('y')
    z = tmp_path / 'out.zip'

    ret = zipx.zip_to_path(str(z), [str(d)])

    assert ret == 2


def test_zip_to_path_empty_directory(tmp_path):
    d = tmp_path / 'empty'
    d.mkdir()
    z = tmp_path / 'out.zip'

    ret = zipx.zip_to_path(str(z), [str(d)])

    assert ret == 0
    assert not z.exists()


def test_zip_to_path_dict_source(tmp_path):
    z = tmp_path / 'out.zip'

    ret = zipx.zip_to_path(str(z), [{'hello.txt': b'world', 'data.bin': b'\x00\x01'}])

    assert ret == 2
    with zipfile.ZipFile(str(z)) as zf:
        assert zf.read('hello.txt') == b'world'
        assert zf.read('data.bin') == b'\x00\x01'


def test_zip_to_path_flat(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    (d / 'f.txt').write_text('flat')
    z = tmp_path / 'out.zip'

    zipx.zip_to_path(str(z), [str(d / 'f.txt')], flat=True)

    with zipfile.ZipFile(str(z)) as zf:
        assert zf.namelist() == ['f.txt']


def test_zip_to_path_basedir(tmp_path):
    d = tmp_path / 'base' / 'sub'
    d.mkdir(parents=True)
    (d / 'f.txt').write_text('hi')
    z = tmp_path / 'out.zip'
    base_dir = str(tmp_path / 'base') + '/'

    zipx.zip_to_path(str(z), [str(d / 'f.txt')], base_dir=base_dir)

    with zipfile.ZipFile(str(z)) as zf:
        assert zf.namelist() == ['sub/f.txt']


def test_zip_to_path_mixed_sources(tmp_path):
    (tmp_path / 'real.txt').write_text('real')
    z = tmp_path / 'out.zip'

    ret = zipx.zip_to_path(str(z), [str(tmp_path / 'real.txt'), {'virtual.txt': b'virtual'}])

    assert ret == 2
    with zipfile.ZipFile(str(z)) as zf:
        assert 'virtual.txt' in zf.namelist()


def test_zip_to_path_invalid_source(tmp_path):
    z = tmp_path / 'out.zip'

    try:
        zipx.zip_to_path(str(z), ['/nonexistent/path/file.txt'])
        assert False, 'expected Error'
    except zipx.Error:
        pass


# --- zip_to_bytes ---

def test_zip_to_bytes_returns_bytes(tmp_path):
    (tmp_path / 'a.txt').write_text('abc')

    result = zipx.zip_to_bytes([str(tmp_path / 'a.txt')])

    assert isinstance(result, bytes)
    assert len(result) > 0
    assert zipfile.is_zipfile(io.BytesIO(result))


def test_zip_to_bytes_empty_sources():
    result = zipx.zip_to_bytes([])

    assert result == b''


def test_zip_to_bytes_dict_source():
    result = zipx.zip_to_bytes([{'note.txt': b'content'}])

    with zipfile.ZipFile(io.BytesIO(result)) as zf:
        assert zf.read('note.txt') == b'content'


def test_zip_to_bytes_flat(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    (d / 'f.txt').write_text('x')

    result = zipx.zip_to_bytes([str(d / 'f.txt')], flat=True)

    with zipfile.ZipFile(io.BytesIO(result)) as zf:
        assert zf.namelist() == ['f.txt']


# --- unzip_path ---

def test_unzip_path_basic(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.txt').write_text('aaa')
    (src / 'b.txt').write_text('bbb')
    z = tmp_path / 'arc.zip'
    zipx.zip_to_path(str(z), [str(src / 'a.txt'), str(src / 'b.txt')], flat=True)

    out = tmp_path / 'out'
    out.mkdir()
    ret = zipx.unzip_path(str(z), str(out), flat=True)

    assert ret == 2
    assert (out / 'a.txt').read_text() == 'aaa'
    assert (out / 'b.txt').read_text() == 'bbb'


def test_unzip_path_preserves_structure(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    (d / 'f.txt').write_text('hi')
    z = tmp_path / 'arc.zip'
    zipx.zip_to_path(str(z), [str(d)], base_dir=str(tmp_path) + '/')

    out = tmp_path / 'out'
    out.mkdir()
    zipx.unzip_path(str(z), str(out), flat=False)

    assert (out / 'sub' / 'f.txt').read_text() == 'hi'


def test_unzip_path_flat(tmp_path):
    d = tmp_path / 'deep' / 'nested'
    d.mkdir(parents=True)
    (d / 'f.txt').write_text('deep')
    z = tmp_path / 'arc.zip'
    zipx.zip_to_path(str(z), [str(d / 'f.txt')])

    out = tmp_path / 'out'
    out.mkdir()
    zipx.unzip_path(str(z), str(out), flat=True)

    assert (out / 'f.txt').read_text() == 'deep'


def test_unzip_path_returns_count(tmp_path):
    files = ['a.txt', 'b.txt', 'c.txt']
    for name in files:
        (tmp_path / name).write_text(name)
    z = tmp_path / 'arc.zip'
    zipx.zip_to_path(str(z), [str(tmp_path / n) for n in files], flat=True)

    out = tmp_path / 'out'
    out.mkdir()
    ret = zipx.unzip_path(str(z), str(out))

    assert ret == 3


# --- unzip_bytes ---

def test_unzip_bytes_basic(tmp_path):
    (tmp_path / 'a.txt').write_text('hello')
    data = zipx.zip_to_bytes([str(tmp_path / 'a.txt')], flat=True)

    out = tmp_path / 'out'
    out.mkdir()
    ret = zipx.unzip_bytes(data, str(out), flat=True)

    assert ret == 1
    assert (out / 'a.txt').read_text() == 'hello'


def test_unzip_bytes_returns_count(tmp_path):
    data = zipx.zip_to_bytes([{'x.txt': b'x', 'y.txt': b'y'}])

    out = tmp_path / 'out'
    out.mkdir()
    ret = zipx.unzip_bytes(data, str(out), flat=True)

    assert ret == 2


# --- unzip_path_to_dict ---

def test_unzip_path_to_dict_flat(tmp_path):
    (tmp_path / 'a.txt').write_text('aaa')
    (tmp_path / 'b.txt').write_text('bbb')
    z = tmp_path / 'arc.zip'
    zipx.zip_to_path(str(z), [str(tmp_path / 'a.txt'), str(tmp_path / 'b.txt')], flat=True)

    result = zipx.unzip_path_to_dict(str(z), flat=True)

    assert result == {'a.txt': b'aaa', 'b.txt': b'bbb'}


def test_unzip_path_to_dict_full_paths(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    (d / 'f.txt').write_text('hi')
    z = tmp_path / 'arc.zip'
    zipx.zip_to_path(str(z), [str(d / 'f.txt')], base_dir=str(tmp_path) + '/')

    result = zipx.unzip_path_to_dict(str(z), flat=False)

    assert result == {'sub/f.txt': b'hi'}


def test_unzip_path_to_dict_no_extraction(tmp_path):
    (tmp_path / 'f.txt').write_text('data')
    z = tmp_path / 'arc.zip'
    zipx.zip_to_path(str(z), [str(tmp_path / 'f.txt')], flat=True)
    (tmp_path / 'f.txt').unlink()

    result = zipx.unzip_path_to_dict(str(z), flat=True)

    assert result == {'f.txt': b'data'}
    assert not (tmp_path / 'f.txt').exists()


# --- unzip_bytes_to_dict ---

def test_unzip_bytes_to_dict(tmp_path):
    data = zipx.zip_to_bytes([{'one.txt': b'1', 'two.txt': b'2'}])

    result = zipx.unzip_bytes_to_dict(data, flat=True)

    assert result == {'one.txt': b'1', 'two.txt': b'2'}


def test_unzip_bytes_to_dict_matches_path_to_dict(tmp_path):
    (tmp_path / 'f.txt').write_text('same')
    z = tmp_path / 'arc.zip'
    zipx.zip_to_path(str(z), [str(tmp_path / 'f.txt')], flat=True)
    data = zipx.zip_to_bytes([str(tmp_path / 'f.txt')], flat=True)

    r1 = zipx.unzip_path_to_dict(str(z), flat=True)
    r2 = zipx.unzip_bytes_to_dict(data, flat=True)

    assert r1 == r2


# --- security: zip slip / malicious entries ---

def test_unzip_skips_absolute_path_entries(tmp_path):
    z = tmp_path / 'evil.zip'
    with zipfile.ZipFile(str(z), 'w') as zf:
        zf.writestr('/etc/passwd', 'evil')
        zf.writestr('safe.txt', 'ok')

    out = tmp_path / 'out'
    out.mkdir()
    ret = zipx.unzip_path(str(z), str(out), flat=True)

    assert ret == 1
    assert (out / 'safe.txt').exists()


def test_unzip_skips_dotdot_entries(tmp_path):
    z = tmp_path / 'evil.zip'
    with zipfile.ZipFile(str(z), 'w') as zf:
        zf.writestr('../escape.txt', 'evil')
        zf.writestr('safe.txt', 'ok')

    out = tmp_path / 'out'
    out.mkdir()
    ret = zipx.unzip_path(str(z), str(out), flat=True)

    assert ret == 1
    assert not (tmp_path / 'escape.txt').exists()


def test_unzip_skips_dot_prefixed_entries(tmp_path):
    z = tmp_path / 'hidden.zip'
    with zipfile.ZipFile(str(z), 'w') as zf:
        zf.writestr('.hidden', 'secret')
        zf.writestr('visible.txt', 'ok')

    out = tmp_path / 'out'
    out.mkdir()
    ret = zipx.unzip_path(str(z), str(out), flat=True)

    assert ret == 1
    assert not (out / '.hidden').exists()


# --- Windows path separators ---

def test_unzip_normalizes_windows_separators(tmp_path):
    z = tmp_path / 'win.zip'
    with zipfile.ZipFile(str(z), 'w') as zf:
        zf.writestr('sub\\file.txt', 'windows')

    result = zipx.unzip_path_to_dict(str(z), flat=True)

    assert result == {'file.txt': b'windows'}
