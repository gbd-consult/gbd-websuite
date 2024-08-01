"""Tests for the serializer module"""

import gws
import gws.test.util as u
import gws.lib.xmlx.serializer

import gws.lib.xmlx as xmlx


def test_to_list():
    el = xmlx.tag('geometry/gml:Point',
                  {'gml:id': 'xy'},
                  ['gml:coordinates', '12.345,56.789'],
                  srsName=3857)
    assert xmlx.serializer.to_list(el) == ['geometry',
                                           [['gml:Point',
                                             {'gml:id': 'xy', 'srsName': 3857},
                                             [['gml:coordinates', '12.345,56.789']]]]]


def test_to_string():
    el = xmlx.tag('geometry/gml:Point',
                  {'gml:id': 'xy'},
                  ['gml:coordinates', '12.345,56.789'],
                  srsName=3857)
    assert xmlx.serializer.to_string(el) == ('<geometry><gml:Point gml:id="xy" '
                                             'srsName="3857"><gml:coordinates>12.345,56.789</gml:coordinates></gml:Point></geometry>')
