"""Tests for the zipx module."""
import io
import os
import zipfile

import gws
import gws.test.util as u
import gws.lib.zipx as zipx


# create new dir with two files, dir1,  dir0 to test zipx
def test_create():
    os.mkdir('/test')
    os.mkdir('/test/dir0')
    os.mkdir('/test/dir1')
    f1 = open('/test/f1.txt', 'x')
    f1.close()
    f2 = open('/test/f2.txt', 'x')
    f2.close()
    f3 = open('/test/dir1/f3.txt', 'x')
    f3.close()
    f4 = open('/test/dir1/f4.txt', 'x')
    f4.close()


def test_zip():
    ret = zipx.zip('/test/zip', '/test/f1.txt', flat=False)
    ret2 = zipx.zip('/test/zip2', '/test/f1.txt', flat=True)
    assert zipfile.is_zipfile('/test/zip')
    assert os.listdir('/test').sort() == ['f1.txt', 'dir1', 'zip', 'zip2', 'dir0', 'f2.txt'].sort()
    assert ret == 1
    assert ret2 == 1
    assert zipx.unzip_to_dict('/test/zip') != zipx.unzip_to_dict('/test/zip2')
    os.remove('/test/zip')
    os.remove('/test/zip2')


# test zip with no src
def test_zip_empty():
    ret = zipx.zip('/test/zip')
    assert os.listdir('/test').sort() == ['f1.txt', 'dir1', 'dir0', 'f2.txt'].sort()
    assert ret == 0


# test dir as src
def test_zip_dir():
    ret = zipx.zip('/test/zip', '/test/dir1')
    assert os.listdir('/test').sort() == ['zip', 'f2.txt', 'f1.txt', 'dir1', 'dir0'].sort()
    assert ret == 2
    os.remove('/test/zip')


# test zip with empty dir
def test_zip_dir_empty():
    ret = zipx.zip('/test/zip', '/test/dir0')
    assert os.listdir('/test').sort() == ['f2.txt', 'f1.txt', 'dir1', 'dir0'].sort()
    assert ret == 0


def test_zip_to_bytes():
    ret = zipx.zip_to_bytes('/test/f1.txt', '/test/f2.txt', flat=False)
    # assert ret = 'someByteString'
    os.remove('/test/f1.txt')
    os.remove('/test/f2.txt')
    zipx.unzip_bytes(ret, '/test', flat=True)
    assert os.listdir('/test').sort() == ['f2.txt', 'f1.txt', 'dir1', 'dir0'].sort()


# test zip_to_bytes with no src
def test_zip_to_bytes_empty():
    ret = zipx.zip_to_bytes()
    assert os.listdir('/test').sort() == ['f2.txt', 'f1.txt', 'dir1', 'dir0'].sort()
    assert ret == b''


# test zip_to_bytes with dir as src
def test_zip_to_bytes_dir():
    ret = zipx.zip_to_bytes('/test/dir1', flat=False)
    # assert ret = 'someByteString'
    zipx.unzip_bytes(ret, '/test', flat=True)
    assert os.listdir('/test').sort() == ['f3.txt', 'f4.txt', 'f2.txt', 'f1.txt', 'dir1', 'dir0'].sort()
    os.remove('/test/f3.txt')
    os.remove('/test/f4.txt')


# test zip_to_bytes with empty dir
def test_zip_to_bytes_dir_empty():
    ret = zipx.zip_to_bytes('/test/dir0')
    assert os.listdir('/test').sort() == ['f2.txt', 'f1.txt', 'dir1', 'dir0'].sort()
    assert ret == b''


# Has no return although return type is int
def test_unzip():
    zipx.zip('/test/zip', '/test/f1.txt', '/test/f2.txt', flat=True)
    os.remove('/test/f1.txt')
    os.remove('/test/f2.txt')
    ret = zipx.unzip('/test/zip', '/test', flat=False)
    os.remove('/test/zip')
    assert os.listdir('/test').sort() == ['f2.txt', 'f1.txt', 'dir1', 'dir0'].sort()
    assert ret == 2


# test for unzipping files into dir containing same named files
def test_unzip_name():
    zipx.zip('/test/zip', '/test/f1.txt', '/test/f2.txt', flat=False)
    os.remove('/test/f1.txt')
    os.remove('/test/f2.txt')
    ret = zipx.unzip('/test/zip', '/test', flat=True)
    os.remove('/test/zip')
    assert os.listdir('/test').sort() == ['f2.txt', 'f1.txt', 'dir1', 'dir0'].sort()
    assert ret == 2


def test_unzip_bytes():
    z = zipx.zip_to_bytes('/test/f1.txt', '/test/f2.txt', flat=False)
    os.remove('/test/f1.txt')
    os.remove('/test/f2.txt')
    ret = zipx.unzip_bytes(z, '/test', flat=True)
    assert ret == 2
    assert os.listdir('/test').sort() == ['f2.txt', 'f1.txt', 'dir1', 'dir0'].sort()


def test_unzip_to_dict():
    zipx.zip('/test/zip', '/test/f1.txt', '/test/f2.txt')
    os.remove('/test/f1.txt')
    os.remove('/test/f2.txt')
    ret = zipx.unzip_to_dict('/test/zip', flat=True)
    os.remove('/test/zip')
    assert ret == {'f1.txt': b'', 'f2.txt': b''}
    assert os.listdir('/test').sort() == ['dir1', 'dir0'].sort()
    f1 = open('/test/f1.txt', 'x')
    f1.close()
    f2 = open('/test/f2.txt', 'x')
    f2.close()


def test_unzip_bytes_to_dict():
    byt = zipx.zip_to_bytes('/test/f1.txt')
    ret = zipx.unzip_bytes_to_dict(byt)
    zipx.zip('/test/zip', '/test/f1.txt')
    ret2 = zipx.unzip_to_dict('/test/zip')
    os.remove('/test/zip')
    assert ret == ret2


# removing test dir and test files
def test_end():
    os.remove('/test/dir1/f3.txt')
    os.remove('/test/dir1/f4.txt')
    os.remove('/test/f1.txt')
    os.remove('/test/f2.txt')
    os.rmdir('/test/dir0')
    os.rmdir('/test/dir1')
    os.rmdir('/test')
