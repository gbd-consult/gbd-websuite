import time
import base64

import _test.util as u
import _test.common.const as cc


def _wait_for_print(job_uid):
    while True:
        res = u.cmd("printerQuery", {"jobUid": job_uid})
        s = res.json()['state']
        if s in ('error', 'cancel'):
            return f'JOB STATE: {s}'
        if s == 'complete':
            return u.req(f'/_/cmd/assetHttpGetResult/jobUid/{job_uid}')
        # print(f'\n> WAITING FOR PRINTER: {s}')
        time.sleep(2)


def test_layers():
    x, y = cc.POINTS.dus1

    params = {
        "type": "template",
        "format": "png",
        "projectUid": "a",
        "templateUid": "html_template",
        "quality": 1,
        "rotation": 0,
        "scale": 2000,
        "items": [
            {
                "type": "layer",
                "layerUid": "a.map.dus1"
            },
            {
                "type": "layer",
                "layerUid": "a.map.dus2"
            },
            {
                "type": "layer",
                "layerUid": "a.map.wms_dus3"
            }
        ],
        "sections": [
            {
                "center": [
                    # left bottom corner should be dus1.x, dus1.y,
                    x + 150,
                    y + 150,
                ],
                "context": {
                    "prop1": "prop_1_value",
                    "prop2": "prop_2_value",
                    "prop3": "prop_3_value_should_be_ignored",
                }
            }
        ]
    }

    r = u.cmd('printerPrint', params).json()
    res = _wait_for_print(r['jobUid'])
    assert u.compare_image_response(res, '/data/layers.png') == ''


def test_features():
    x, y = cc.POINTS.dus1

    params = {
        "type": "template",
        "format": "png",
        "projectUid": "a",
        "templateUid": "html_template",
        "quality": 1,
        "rotation": 0,
        "scale": 2000,
        "items": [
            {
                "type": "features",
                "features": [
                    {
                        "elements": {"label": "point"},
                        "shape": {
                            "crs": cc.CRS_3857,
                            "geometry": {
                                "type": "Point",
                                "coordinates": [x + 50, y + 250],
                            }
                        },
                        "style": {
                            "type": "css",
                            "values": {
                                "fill": "rgb(0,255,255)",
                                "point_size": 20,
                                "label_font_size": 12,
                                "label_fill": "white",
                                "label_background": "blue",
                                "label_placement": "start",
                                "label_offset_y": 20,
                            }
                        }
                    },
                    {
                        "elements": {"label": "polygon"},
                        "shape": {
                            "crs": cc.CRS_3857,
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [x, y],
                                    [x + 100, y + 100],
                                    [x + 100, y + 200],
                                    [x, y],
                                ]]
                            }
                        },
                        "style": {
                            "type": "css",
                            "values": {
                                "fill": "rgba(255,0,0,0.5)",
                                "stroke": "rgba(0,255,0.5)",
                                "stroke_width": 3,
                                "label_font_size": 13,
                                "label_fill": "white",
                                "label_background": "blue",
                            }
                        }
                    },
                    {
                        "elements": {"label": "polygon-cropped"},
                        "shape": {
                            "crs": cc.CRS_3857,
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [x + 200, y + 200],
                                    [x + 200, y + 500],
                                    [x + 400, y + 500],
                                    [x + 200, y + 200],
                                ]]
                            }
                        },
                        "style": {
                            "type": "css",
                            "values": {
                                "fill": "rgba(0,255,0,0.5)",
                            }
                        }
                    }
                ]
            },
            {
                "type": "layer",
                "layerUid": "a.map.dus1"
            }
        ],
        "sections": [
            {
                "center": [
                    # left bottom corner should be dus1.x, dus1.y,
                    x + 150,
                    y + 150,
                ],
                "context": {
                    "prop1": "prop_1_value",
                    "prop2": "prop_2_value",
                    "prop3": "prop_3_value_should_be_ignored",
                }
            }
        ]
    }

    r = u.cmd('printerPrint', params).json()
    res = _wait_for_print(r['jobUid'])
    assert u.compare_image_response(res, '/data/features.png') == ''


def test_bitmap():
    x, y = cc.POINTS.dus1

    img = u.read('/data/chess.png', 'rb')
    img = base64.encodebytes(img).decode('utf8')
    img = 'data:image/png;base64,' + img

    params = {
        "type": "template",
        "format": "png",
        "projectUid": "a",
        "templateUid": "html_template",
        "quality": 1,
        "rotation": 0,
        "scale": 2000,
        "items": [
            {
                "type": "layer",
                "layerUid": "a.map.dus1"
            },
            {
                "type": "url",
                "url": img
            }
        ],
        "sections": [
            {
                "center": [
                    # left bottom corner should be dus1.x, dus1.y,
                    x + 150,
                    y + 150,
                ]
            }
        ]
    }

    r = u.cmd('printerPrint', params).json()
    res = _wait_for_print(r['jobUid'])
    assert u.compare_image_response(res, '/data/bitmap.png') == ''


def test_qgis_template():
    x, y = cc.POINTS.dus1

    params = {
        "type": "template",
        "format": "png",
        "projectUid": "a",
        "templateUid": "qgis_template",
        "quality": 1,
        "rotation": 0,
        "scale": 2000,
        "items": [
            {
                "type": "layer",
                "layerUid": "a.map.dus1"
            },
            {
                "type": "layer",
                "layerUid": "a.map.dus2"
            },
            {
                "type": "layer",
                "layerUid": "a.map.wms_dus3"
            }
        ],
        "sections": [
            {
                "center": [
                    # left bottom corner should be dus1.x, dus1.y,
                    x + 150,
                    y + 150,
                ],
                "context": {
                    "prop1": "prop_1_value",
                    "prop2": "prop_2_value",
                    "prop3": "prop_3_value_should_be_ignored",
                }
            }
        ]
    }

    r = u.cmd('printerPrint', params).json()
    res = _wait_for_print(r['jobUid'])
    assert u.compare_image_response(res, '/data/qgis.png') == ''

