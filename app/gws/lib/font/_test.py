"""Tests for the font module."""

from unittest import mock

import gws
import gws.lib.font as font
import gws.test.util as u


def test_configure():
    # Test with no directory
    config = font.Config(dir=None)
    with mock.patch('gws.lib.font.install_fonts') as mock_install:
        font.configure(config)
        mock_install.assert_not_called()

        # Test with directory
    config = font.Config(dir="/test/fonts")
    with mock.patch('gws.lib.font.install_fonts') as mock_install:
        font.configure(config)
        mock_install.assert_called_once_with("/test/fonts")


def test_install_fonts():
    with mock.patch('gws.lib.osx.run') as mock_run, \
            mock.patch('gws.lib.osx.find_files', return_value=['/test/fonts/font1.ttf', '/test/fonts/font2.ttf']):
        font.install_fonts('/test/fonts')

        # Check mkdir call
        mock_run.assert_any_call(['mkdir', '-p', '/usr/local/share/fonts'], echo=True)

        # Check copy calls
        mock_run.assert_any_call(['cp', '-v', '/test/fonts/font1.ttf', '/usr/local/share/fonts'], echo=True)
        mock_run.assert_any_call(['cp', '-v', '/test/fonts/font2.ttf', '/usr/local/share/fonts'], echo=True)

        # Check font cache update
        mock_run.assert_any_call(['fc-cache', '-fv'], echo=True)


def test_from_name():
    # Mock the PIL.ImageFont.truetype function
    with mock.patch('PIL.ImageFont.truetype') as mock_truetype:
        mock_font = mock.MagicMock()
        mock_truetype.return_value = mock_font

        result = font.from_name("Arial", 12)

        # Check that truetype was called with correct parameters
        mock_truetype.assert_called_once_with(font="Arial", size=12)
        assert result == mock_font
