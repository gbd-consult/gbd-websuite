"""Test utilities."""

import typing

import pytest

import gws.lib.cli

from . import auth, check, http, log, metadata, model, pg
from . import mockserver_client as mockserver
from .config import gws_root, gws_specs
from .options import option, load_options

##

fixture = pytest.fixture
raises = pytest.raises

cast = typing.cast

exec = gws.lib.cli.exec
