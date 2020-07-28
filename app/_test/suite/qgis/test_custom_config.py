import _test.util as u
import _test.common.const as cc


def test_custom_config_single_layer():
    r = u.cmd('mapDescribeLayer', {
        'layerUid': 'custom_config.map.points_ghana_25832',
    })

    assert r.json() == {'description': 'points_ghana_25832 custom description'}


def test_custom_config_for_multiple_layers():
    r = u.cmd('mapRenderLegend', {
        'layerUid': 'custom_config.map.squares_ny_2263',
    })

    a, b = u.compare_image(r, '/data/response_images/custom_legend_1.png')
    assert a == b

    r = u.cmd('mapRenderLegend', {
        'layerUid': 'custom_config.map.squares_dus1_3857',
    })

    a, b = u.compare_image(r, '/data/response_images/custom_legend_1.png')
    assert a == b


def test_custom_config_override():

    r = u.cmd('mapRenderLegend', {
        'layerUid': 'custom_config.map.squares_dus2_3857',
    })

    a, b = u.compare_image(r, '/data/response_images/custom_legend_2.png')
    assert a == b
