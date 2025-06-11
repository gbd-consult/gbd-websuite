"""Tests for the inspire module."""

import gws
import gws.base.metadata.inspire as inspire
import gws.test.util as u


def test_theme_name():
    assert inspire.theme_name('ac', 'de') == 'Atmosphärische Bedingungen'
    assert inspire.theme_name('ac', 'en') == 'Atmospheric conditions'


def test_theme_definition():
    assert inspire.theme_definition('ac',
                                    'en') == 'Physical conditions in the atmosphere. Includes spatial data based on measurements, on models or on a combination thereof and includes measurement locations.'
    assert inspire.theme_definition('ac',
                                    'de') == 'Physikalische Bedingungen in der Atmosphäre. Dazu zählen Geodaten auf der Grundlage von Messungen, Modellen oder einer Kombination aus beiden sowie Angabe der Messstandorte.'
