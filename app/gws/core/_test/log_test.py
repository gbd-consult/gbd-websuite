"""Tests for the log module."""

import unittest.mock as mock

import gws


def test_exception_without_active_exception():
    with mock.patch.object(gws.log, '_raw') as raw:
        gws.log.exception()
    raw.assert_called_once()
    assert raw.call_args.args[0] == gws.log.Level.ERROR
    assert raw.call_args.args[1] == 'exception() called without an active exception'


def test_exception_without_active_exception_with_message():
    with mock.patch.object(gws.log, '_raw') as raw:
        gws.log.exception('message_1')
    raw.assert_called_once()
    assert raw.call_args.args[1] == 'message_1'


def test_exception_with_active_exception():
    with mock.patch.object(gws.log, '_raw') as raw:
        try:
            raise ValueError('error_1')
        except ValueError:
            gws.log.exception()
    assert 'error_1' in raw.call_args_list[0].args[1]
