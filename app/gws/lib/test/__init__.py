"""Support library for tests."""

from . import (
    config,
    features,
    glob,
    http,
    mockserv,
    postgres,
    util,
)

from .util import setup, teardown, sleep

import pytest

fixture = pytest.fixture


def raises(exc=None):
    return pytest.raises(exc or Exception)
