"""Tests for the inifile module."""
import os

import gws
import gws.test.util as u
import gws.lib.inifile as inifile


def test_from_paths():
    with open('/tmp/myfile.ini', 'w') as f:
        f.write('[mysection]\n'
                'mykey = myval\n'
                'morekeys = morevals\n'
                '\n'
                '[mysection.subsection]\n'
                'anotherkey = 123\n')
        f.close()
        p1 = '/tmp/myfile.ini'
        assert inifile.from_paths(p1) == {'mysection.mykey': 'myval',
                                          'mysection.morekeys': 'morevals',
                                          'mysection.subsection.anotherkey': '123'}


def test_from_paths_empty():
    assert inifile.from_paths() == {}


def test_to_string_empty():
    assert inifile.to_string({}) == ''


def test_to_string():
    d = {'de.modSidebarOpenButton': 'Menü öffnen',
         'en.modSidebarOpenButton': 'open sidebar',
         }
    assert inifile.to_string(d).replace('\n', '') == ('[de]'
                                                      'modsidebaropenbutton=Menü öffnen'
                                                      '[en]'
                                                      'modsidebaropenbutton=open sidebar')
