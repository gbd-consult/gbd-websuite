"""Tests for the legend module."""

import gws
import gws.test.util as u
import gws.base.legend as legend
import gws.lib.image as image
import unittest.mock as mock


def test_output_to_bytes(tmp_path):
    img = image.from_size((50, 50), color=(255, 0, 0, 255))
    img.to_path(str(tmp_path / 'img.png'))
    lro = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'img.png')}' alt='image'>",
        image=img,
        image_path=str(tmp_path / 'img.png'),
        size=gws.Size((50.0, 50.5)),
        mime="image/jpeg"
    )
    assert legend.output_to_bytes(lro) == img.to_bytes()


def test_output_to_bytes_none(tmp_path):
    lro = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'img.png')}' alt='image'>",
        image=None,
        image_path=None,
        size=gws.Size((50.0, 50.5)),
        mime="image/jpeg"
    )
    assert not legend.output_to_bytes(lro)


def test_output_to_image(tmp_path):
    img = image.from_size((50, 50), color=(255, 0, 0, 255))
    img.to_path(str(tmp_path / 'img.png'))
    lro = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'img.png')}' alt='image'>",
        image=img,
        image_path=str(tmp_path / 'img.png'),
        size=gws.Size((50.0, 50.5)),
        mime="image/jpeg"
    )
    assert legend.output_to_image(lro).compare_to(img) == 0


def test_output_to_image_from_path(tmp_path):
    img = image.from_size((50, 50), color=(255, 0, 0, 255))
    img.to_path(str(tmp_path / 'img.png'))
    lro = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'img.png')}' alt='image'>",
        image=None,
        image_path=str(tmp_path / 'img.png'),
        size=gws.Size((50.0, 50.5)),
        mime="image/jpeg"
    )
    assert legend.output_to_image(lro).compare_to(img) == 0


def test_output_to_image_none(tmp_path):
    lro = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'img.png')}' alt='image'>",
        image=None,
        image_path=None,
        size=gws.Size((50.0, 50.5)),
        mime="image/jpeg"
    )
    assert not legend.output_to_image(lro)


# used mock to avoid randomized naming of the image path
def test_output_to_image_path(tmp_path):
    with mock.patch('gws.u.ephemeral_path', return_value=str(tmp_path / 'test_legend.png')):
        img = image.from_size((50, 50), color=(255, 0, 0, 255))
        img.to_path(str(tmp_path / 'img.png'))
        lro = gws.LegendRenderOutput(
            html=f"<img src='{str(tmp_path / 'img.png')}' alt='image'>",
            image=img,
            image_path=str(tmp_path / 'img.png'),
            size=gws.Size((50.0, 50.5)),
            mime="image/jpeg"
        )
        assert legend.output_to_image_path(lro) == str(tmp_path / 'test_legend.png')
        assert image.from_path(legend.output_to_image_path(lro)).compare_to(img) == 0


def test_combine_outputs(tmp_path):
    img = image.from_size((5, 10), color=(255, 255, 255, 255))
    red = image.from_size((5, 5), color=(255, 0, 0, 255))
    blue = image.from_size((5, 5), color=(0, 0, 255, 255))
    img.paste(red, where=(0, 0))
    img.paste(blue, where=(0, 5))
    img.to_path(str(tmp_path / 'combined.png'))

    red.to_path(str(tmp_path / 'red.png'))
    blue.to_path(str(tmp_path / 'blue.png'))

    lro_1 = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'red.png')}' alt='red'>",
        image=red,
        image_path=str(tmp_path / 'red.png'),
        size=gws.Size((5.0, 5.0)),
        mime="image/jpeg"
    )

    lro_2 = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'blue.png')}' alt='blue'>",
        image=blue,
        image_path=str(tmp_path / 'blue.png'),
        size=gws.Size((5.0, 5.0)),
        mime="image/jpeg"
    )
    assert legend.combine_outputs([lro_1, lro_2]).image.compare_to(img) == 0


def test_combine_outputs_none(tmp_path):
    lro_1 = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'red.png')}' alt='red'>",
        image=None,
        image_path=None,
        size=gws.Size((5.0, 5.0)),
        mime="image/jpeg"
    )

    lro_2 = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'blue.png')}' alt='blue'>",
        image=None,
        image_path=None,
        size=gws.Size((5.0, 5.0)),
        mime="image/jpeg"
    )
    assert not legend.combine_outputs([lro_1, lro_2])
