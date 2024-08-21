"""Tests for the zipx module."""
import io
import os
import zipfile

import gws
import gws.test.util as u
import gws.lib.zipx as zipx


def test_zip(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    z = d / 'zip'
    z2 = d / 'zip2'
    f = d / 'f1.txt'
    f.write_text('foobar')

    ret = zipx.zip(str(z), str(f), flat=False)
    ret2 = zipx.zip(str(z2), str(f), flat=True)
    assert zipfile.is_zipfile(z)
    assert sorted(os.listdir(d)) == sorted(['zip', 'zip2', 'f1.txt'])
    assert ret == 1
    assert ret2 == 1
    assert zipx.unzip_to_dict(str(z)) != zipx.unzip_to_dict(str(z2))


# test zip with no src
def test_zip_empty(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()
    z = d / 'zip'
    ret = zipx.zip(str(z))
    assert not os.listdir(str(d))
    assert ret == 0


# test dir as src
def test_zip_dir(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    z = d / 'zip'
    dr = d / 'dir'
    dr.mkdir()

    f = dr / 'f1.txt'
    f.write_text('foo')
    f2 = dr / 'f2.txt'
    f2.write_text('bar')

    ret = zipx.zip(str(z), str(dr))
    assert sorted(os.listdir(str(d))) == sorted(['zip', 'dir'])
    assert ret == 2


# test zip with empty dir
def test_zip_dir_empty(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    z = d / 'zip'
    dr = d / 'dir'
    dr.mkdir()

    ret = zipx.zip(str(z), str(dr))
    assert sorted(os.listdir(str(d))) == sorted(['dir'])
    assert ret == 0


def test_zip_to_bytes(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    f = d / 'f1.txt'
    f.write_text('foo')
    f2 = d / 'f2.txt'
    f2.write_text('bar')

    ret = zipx.zip_to_bytes(str(f), str(f2), flat=False)
    # assert ret = 'someByteString'
    os.remove(str(f))
    os.remove(str(f2))
    zipx.unzip_bytes(ret, str(d), flat=True)
    assert sorted(os.listdir(str(d))) == sorted(['f2.txt', 'f1.txt'])


# test zip_to_bytes with no src
def test_zip_to_bytes_empty(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    ret = zipx.zip_to_bytes()
    assert not os.listdir(str(d))
    assert not sorted(os.listdir(d))
    assert ret == b''


# test zip_to_bytes with dir as src
def test_zip_to_bytes_dir(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    z = d / 'zip'
    dr = d / 'dir'
    dr.mkdir()

    f = dr / 'f1.txt'
    f.write_text('foo')
    f2 = dr / 'f2.txt'
    f2.write_text('bar')

    ret = zipx.zip_to_bytes(str(dr), flat=False)
    # assert ret = 'someByteString'
    zipx.unzip_bytes(ret, str(d), flat=True)
    assert sorted(os.listdir(str(d))) == sorted(['f1.txt', 'f2.txt', 'dir'])


# test zip_to_bytes with empty dir
def test_zip_to_bytes_dir_empty(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    dr = d / 'dir'
    dr.mkdir()

    ret = zipx.zip_to_bytes(str(dr))
    assert sorted(os.listdir(d)) == sorted(['dir'])
    assert ret == b''


# Has no return although return type is int
def test_unzip(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    z = d / 'zip'

    f = d / 'f1.txt'
    f.write_text('foo')
    f2 = d / 'f2.txt'
    f2.write_text('bar')

    zipx.zip(str(z), str(f), str(f2), flat=True)
    os.remove(str(f))
    os.remove(str(f2))
    ret = zipx.unzip(str(z), str(d), flat=False)
    os.remove(str(z))
    assert sorted(os.listdir(str(d))) == sorted(['f2.txt', 'f1.txt'])
    assert ret == 2


# test for unzipping files into dir containing same named files
def test_unzip_name(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    z = d / 'zip'

    f = d / 'f1.txt'
    f.write_text('foo')
    f2 = d / 'f2.txt'
    f2.write_text('bar')

    zipx.zip(str(z), str(f), str(f2), flat=False)
    os.remove(str(f))
    os.remove(str(f2))
    ret = zipx.unzip(str(z), str(d), flat=True)
    os.remove(str(z))
    assert sorted(os.listdir(str(d))) == sorted(['f2.txt', 'f1.txt'])
    assert ret == 2


def test_unzip_bytes(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    z = d / 'zip'

    f = d / 'f1.txt'
    f.write_text('foo')
    f2 = d / 'f2.txt'
    f2.write_text('bar')

    z = zipx.zip_to_bytes(str(f), str(f2), flat=False)
    os.remove(str(f))
    os.remove(str(f2))
    ret = zipx.unzip_bytes(z, str(d), flat=True)
    assert ret == 2
    assert sorted(os.listdir(str(d))) == sorted(['f1.txt', 'f2.txt'])


def test_unzip_to_dict(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    z = d / 'zip'

    f = d / 'f1.txt'
    f.write_text('foo')
    f2 = d / 'f2.txt'
    f2.write_text('bar')

    zipx.zip(str(z), str(f), str(f2))
    os.remove(str(f))
    os.remove(str(f2))
    ret = zipx.unzip_to_dict(str(z), flat=True)
    assert ret == {'f1.txt': b'foo', 'f2.txt': b'bar'}
    assert sorted(os.listdir(str(d))) == sorted(['zip'])


def test_unzip_bytes_to_dict(tmp_path):
    d = tmp_path / 'test'
    d.mkdir()

    z = d / 'zip'

    f = d / 'f1.txt'
    f.write_text('foo')
    byt = zipx.zip_to_bytes(str(f))
    ret = zipx.unzip_bytes_to_dict(byt)
    zipx.zip(str(z), str(f))
    os.remove(str(f))
    ret2 = zipx.unzip_to_dict(str(z))
    assert ret == ret2
    assert sorted(os.listdir(d)) == ['zip']
