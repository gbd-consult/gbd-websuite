"""Tests for the csv module."""

import gws
import gws.test.util as u
import gws.lib.intl as intl
import gws.plugin.csv_helper as csv


def test_write_headers():
    cfg = csv.Config(format=gws.Config(delimiter=",", encoding='utf8', formulaHack=True, quote='"', quoteAll=False, rowDelimiter='\n'))
    obj = csv.Object()
    obj.initialize(cfg)
    wrt = obj.writer(intl.locale('en_US'))
    wrt.write_headers(['foo', 'bar', 'foo'])
    assert wrt.headers == r'"foo","bar","foo"'


def test_write_row():
    cfg = csv.Config(format=gws.Config(delimiter=",", encoding='utf8', formulaHack=True, quote='"', quoteAll=False, rowDelimiter='\n'))
    obj = csv.Object()
    obj.initialize(cfg)
    wrt = obj.writer(intl.locale('en_US'))
    wrt.write_row(['foo', 'bar', 'foo'])
    assert wrt.rows == ['"foo","bar","foo"']


def test_to_str():
    cfg = csv.Config(format=gws.Config(delimiter=",", encoding='utf8', formulaHack=True, quote='"', quoteAll=False, rowDelimiter='\n'))
    obj = csv.Object()
    obj.initialize(cfg)
    wrt = obj.writer(intl.locale('en_US'))
    wrt.write_headers(['h1', 'h2', 'h3'])
    wrt.write_row(['r1', 'r2'])
    wrt.write_row(['r3', 'r4'])
    assert wrt.to_str() == (
        '"h1","h2","h3"\n'
        '"r1","r2"\n'
        '"r3","r4"'
    )


def test_to_bytes():
    cfg = csv.Config(format=gws.Config(delimiter=",", encoding='utf8', formulaHack=True, quote='"', quoteAll=False, rowDelimiter='\n'))
    obj = csv.Object()
    obj.initialize(cfg)
    wrt = obj.writer(intl.locale('en_US'))
    wrt.write_headers(['h1', 'h2', 'h3'])
    wrt.write_row(['r1', 'r2'])
    assert wrt.to_bytes('utf8').decode('utf8') == wrt.to_str()
    assert wrt.to_bytes('utf8') == wrt.to_str().encode('utf8')
