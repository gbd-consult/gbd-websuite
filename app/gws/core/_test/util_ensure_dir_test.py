"""Tests for ensure_dir."""

import os
import unittest.mock as mock

import gws


def test_creates_nested_dirs(tmp_path):
    path = str(tmp_path / 'dir_1' / 'dir_2')
    assert gws.u.ensure_dir(path) == path
    assert os.path.isdir(path)


def test_tolerates_concurrent_creation(tmp_path):
    path = str(tmp_path / 'dir_1')
    real_mkdir = os.mkdir

    def racing_mkdir(p, mode):
        real_mkdir(p, mode)
        real_mkdir(p, mode)

    with mock.patch('os.mkdir', side_effect=racing_mkdir):
        assert gws.u.ensure_dir(path) == path
    assert os.path.isdir(path)
