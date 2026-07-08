"""Tests for the inifile module."""

import gws.lib.inifile as inifile

_INI_A = """\
[database]
host=localhost
port=5432

[server]
host=0.0.0.0
port=8080
"""

_INI_B = """\
[database]
name=mydb
user=admin

[logging]
level=INFO
"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# --- from_paths ---

def test_from_paths_single_file(tmp_path):
    p = _write(tmp_path, 'a.ini', _INI_A)
    result = inifile.from_paths(p)
    assert result == {
        'database': {'host': 'localhost', 'port': '5432'},
        'server': {'host': '0.0.0.0', 'port': '8080'},
    }


def test_from_paths_multiple_files_merge(tmp_path):
    pa = _write(tmp_path, 'a.ini', _INI_A)
    pb = _write(tmp_path, 'b.ini', _INI_B)
    result = inifile.from_paths(pa, pb)
    assert result['database'] == {'host': 'localhost', 'port': '5432', 'name': 'mydb', 'user': 'admin'}
    assert result['server'] == {'host': '0.0.0.0', 'port': '8080'}
    assert result['logging'] == {'level': 'INFO'}


def test_from_paths_later_file_overwrites(tmp_path):
    ini1 = '[section]\nkey=first\n'
    ini2 = '[section]\nkey=second\n'
    p1 = _write(tmp_path, 'first.ini', ini1)
    p2 = _write(tmp_path, 'second.ini', ini2)
    result = inifile.from_paths(p1, p2)
    assert result['section']['key'] == 'second'


def test_from_paths_no_files():
    result = inifile.from_paths()
    assert result == {}


def test_from_paths_missing_file_is_ignored(tmp_path):
    p = _write(tmp_path, 'real.ini', '[sec]\nk=v\n')
    result = inifile.from_paths(p, '/nonexistent/path.ini')
    assert result == {'sec': {'k': 'v'}}


def test_from_paths_preserves_key_case(tmp_path):
    p = _write(tmp_path, 'a.ini', '[Section]\nMyKey=MyValue\n')
    result = inifile.from_paths(p)
    assert 'MyKey' in result['Section']
    assert result['Section']['MyKey'] == 'MyValue'


# --- from_paths_flat ---

def test_from_paths_flat_single_file(tmp_path):
    p = _write(tmp_path, 'a.ini', _INI_A)
    result = inifile.from_paths_flat(p)
    assert result == {
        'database.host': 'localhost',
        'database.port': '5432',
        'server.host': '0.0.0.0',
        'server.port': '8080',
    }


def test_from_paths_flat_multiple_files(tmp_path):
    pa = _write(tmp_path, 'a.ini', _INI_A)
    pb = _write(tmp_path, 'b.ini', _INI_B)
    result = inifile.from_paths_flat(pa, pb)
    assert result['database.host'] == 'localhost'
    assert result['database.name'] == 'mydb'
    assert result['logging.level'] == 'INFO'


def test_from_paths_flat_later_file_overwrites(tmp_path):
    ini1 = '[sec]\nkey=original\n'
    ini2 = '[sec]\nkey=overridden\n'
    p1 = _write(tmp_path, '1.ini', ini1)
    p2 = _write(tmp_path, '2.ini', ini2)
    result = inifile.from_paths_flat(p1, p2)
    assert result['sec.key'] == 'overridden'


def test_from_paths_flat_no_files():
    result = inifile.from_paths_flat()
    assert result == {}


def test_from_paths_flat_preserves_key_case(tmp_path):
    p = _write(tmp_path, 'a.ini', '[Sec]\nMyKey=val\n')
    result = inifile.from_paths_flat(p)
    assert 'Sec.MyKey' in result
    assert result['Sec.MyKey'] == 'val'


# --- to_string ---

def test_to_string_basic():
    d = {'db.host': 'localhost', 'db.port': '5432'}
    s = inifile.to_string(d)
    assert '[db]' in s
    assert 'host=localhost' in s
    assert 'port=5432' in s


def test_to_string_multiple_sections():
    d = {
        'alpha.key': 'a',
        'beta.key': 'b',
    }
    s = inifile.to_string(d)
    assert '[alpha]' in s
    assert '[beta]' in s
    assert 'key=a' in s
    assert 'key=b' in s


def test_to_string_empty_dict():
    s = inifile.to_string({})
    assert s.strip() == ''


def test_to_string_roundtrip(tmp_path):
    original = {
        'database.host': 'localhost',
        'database.port': '5432',
        'server.debug': 'true',
    }
    s = inifile.to_string(original)
    p = _write(tmp_path, 'out.ini', s)
    result = inifile.from_paths_flat(p)
    assert result == original
