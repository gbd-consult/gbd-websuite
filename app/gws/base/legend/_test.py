"""Tests for the legend module."""

import gws
import gws.test.util as u
import gws.base.legend as legend
import gws.lib.image as image


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
        size=gws.Size((5.0, 5.0))
    )

    lro_2 = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'blue.png')}' alt='blue'>",
        image=blue,
        size=gws.Size((5.0, 5.0))
    )
    assert legend.combine_outputs([lro_1, lro_2]).image.compare_to(img) == 0


def test_combine_outputs_none(tmp_path):
    lro_1 = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'red.png')}' alt='red'>",
        image=None,
        size=gws.Size((5.0, 5.0))
    )

    lro_2 = gws.LegendRenderOutput(
        html=f"<img src='{str(tmp_path / 'blue.png')}' alt='blue'>",
        image=None,
        size=gws.Size((5.0, 5.0))
    )
    assert not legend.combine_outputs([lro_1, lro_2])
