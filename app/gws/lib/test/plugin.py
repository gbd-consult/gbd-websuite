"""Global test fixtures."""

import pytest

import gws
from . import util


@pytest.fixture(scope='module', autouse=True)
def configuration(request):
    gws.log.debug(f'TEST:SETUP:{request.path}')
    util.setup()
    yield
    gws.log.debug(f'TEST:TEARDOWN:{request.path}')
    util.teardown()
