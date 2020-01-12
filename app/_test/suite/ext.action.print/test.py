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


def xxxtest_layers():
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
                "attributes": [
                    {
                        "name": "prop1",
                        "value": "prop_1_value"
                    },
                    {
                        "name": "prop2",
                        "value": "prop_2_value"
                    },
                    {
                        "name": "prop3",
                        "value": "prop_3_value_should_be_ignored"
                    }
                ]
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
                            "content": {
                                "fill": "rgb(0,255,255)",
                                "point-size": "20px",
                                "label-font-size": "12px",
                                "label-fill": "white",
                                "label-background": "blue",
                                "label-placement": "start",
                                "label-offset-y": "20px",
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
                            "content": {
                                "fill": "rgba(255,0,0,0.5)",
                                "stroke": "rgba(0,255,0.5)",
                                "stroke-width": "3pm",
                                "label-font-size": "13px",
                                "label-fill": "white",
                                "label-background": "blue",
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
                            "content": {
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
