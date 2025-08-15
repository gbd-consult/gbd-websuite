"""Tests for the MapServer module."""
import gws
import gws.gis.ms as ms
from gws.test import util as u
import gws.lib.crs as crs
import gws.lib.image as image


def create_cross():
    img = image.from_size((12, 12), color=(255, 255, 255, 255))

    block = image.from_size((4, 12), color=(0, 0, 0, 255))
    img.paste(block, where=(4, 0))

    block = image.from_size((12, 4), color=(0, 0, 0, 255))
    img.paste(block, where=(0, 4))

    output_path = f"/data/cross.png"
    img.to_path(str(output_path))
    return output_path


def create_square(color, size, crs=None, pgw_str=None):
    if color == "red":
        c = (255, 0, 0, 125)
    elif color == "blue":
        c = (0, 0, 255, 125)
    else:
        c = (0, 255, 0, 125)
    img = image.from_size((size, size), color=(255, 255, 255, 255))

    block = image.from_size((size, size), color=c)

    img.paste(block, where=(0, 0))

    output_path = f"/data/{color}_square_{crs}.png"
    img.to_path(str(output_path))

    if crs:
        with open(f"/data/{color}_square_{crs}.pgw", "w") as file:
            file.write(pgw_str)

        with open(f"/data/{color}_square_{crs}.prj", "w") as file:
            file.write(f"""EPSG:{crs}""")
    return output_path


def red_square_4326():
    return create_square("red", 100, 4326, "0.01\n0\n0\n-0.01\n5.005\n54.995")


def blue_square_4326():
    return create_square("blue", 100, 4326, "0.01\n0\n0\n-0.01\n5.005\n54.995")


def red_square_3857():
    return create_square("red", 100, 3857, "1113.194908\n0\n0\n-1113.194908\n557154.0514203343\n7360895.77546744")


def blue_square_3857():
    return create_square("blue", 100, 3857,
                         "1113.194908\n0\n0\n-1113.194908\n557154.0514203343\n7360895.77546744")


def test_rendering():
    raster_image_path = create_square("red", 100)

    map = ms.new_map()

    map.add_layer(
        ms.LayerOptions(type=ms.LayerType.raster,
                        path=str(raster_image_path),
                        crs=crs.WEBMERCATOR
                        )
    )

    img = map.draw(
        bounds=gws.Bounds(
            extent=[0, 0, 100, 100],
            crs=crs.WEBMERCATOR,
        ),
        size=(100, 100),
    )
    image_original = image.from_path("/gws-app/gws/gis/ms/_test/red_square.png")

    assert img.compare_to(image_original) < 0.01


def test_rendering_add_layer():
    raster_image_path = create_square("red", 100)

    map = ms.new_map()

    map.add_layer_from_config(f'''
                    LAYER
                        TYPE RASTER
                        STATUS ON
                        DATA "{raster_image_path}"
                        PROJECTION
                            "init=epsg:3857"
                        END
                    END
                ''')

    img = map.draw(
        bounds=gws.Bounds(
            extent=[0, 0, 100, 100],
            crs=crs.WEBMERCATOR,
        ),
        size=(100, 100),
    )

    image_original = image.from_path("/gws-app/gws/gis/ms/_test/red_square.png")

    assert img.compare_to(image_original) < 0.01


def test_reprojecting():
    map = ms.new_map()

    # adjust image
    rsl_path = blue_square_4326()
    rsl_opt2 = ms.LayerOptions(type=ms.LayerType.raster, path=str(rsl_path), crs=crs.WGS84)
    map.add_layer(rsl_opt2)

    # adjust image
    rsl_path = red_square_3857()
    rsl_opt = ms.LayerOptions(type=ms.LayerType.raster, path=str(rsl_path), crs=crs.WEBMERCATOR)
    map.add_layer(rsl_opt)

    img = map.draw(
        bounds=gws.Bounds(
            extent=[(5.005 - 1) + 1, (54.995 - 1) - 1, (5.005 + 1) + 1, (54.995 + 1) - 1],
            crs=crs.WGS84,
        ),
        size=(200, 200),
    )

    image_original = image.from_path("/gws-app/gws/gis/ms/_test/overlay.png")

    assert img.compare_to(image_original) < 0.01


def test_reprojecting_bottom_right():
    map = ms.new_map()

    # adjust image
    rsl_path = blue_square_4326()
    rsl_opt2 = ms.LayerOptions(type=ms.LayerType.raster, path=str(rsl_path), crs=crs.WGS84)
    map.add_layer(rsl_opt2)

    # adjust image
    rsl_path = red_square_3857()
    rsl_opt = ms.LayerOptions(type=ms.LayerType.raster, path=str(rsl_path), crs=crs.WEBMERCATOR)
    map.add_layer(rsl_opt)

    img = map.draw(
        bounds=gws.Bounds(
            extent=[(5.005 - 1), (54.995 - 1), (5.005 + 1), (54.995 + 1)],
            crs=crs.WGS84,
        ),
        size=(200, 200),
    )

    image_original = image.from_path("/gws-app/gws/gis/ms/_test/overlay_bottom_right.png")

    assert img.compare_to(image_original) < 0.01


def test_reprojecting_crs():
    map = ms.new_map()

    # adjust image
    rsl_path = blue_square_3857()
    rsl_opt2 = ms.LayerOptions(type=ms.LayerType.raster, path=str(rsl_path), crs=crs.WEBMERCATOR)
    map.add_layer(rsl_opt2)

    # adjust image
    rsl_path = red_square_4326()
    rsl_opt = ms.LayerOptions(type=ms.LayerType.raster, path=str(rsl_path), crs=crs.WGS84)
    map.add_layer(rsl_opt)

    img = map.draw(
        bounds=gws.Bounds(
            extent=[(557154.0514203343 - 111319.4908) + 111319.4908, (7360895.77546744 - 111319.4908) - 111319.4908,
                    (557154.0514203343 + 111319.4908) + 111319.4908, (7360895.77546744 + 111319.4908) - 111319.4908],
            # 557154.0514203343\n7360895.77546744
            crs=crs.WEBMERCATOR,
        ),
        size=(200, 200),
    )
    image_original = image.from_path("/gws-app/gws/gis/ms/_test/reprojecting_crs.png")

    assert img.compare_to(image_original) < 0.01


def test_vectors():
    map = ms.new_map()

    map.add_layer_from_config('''
                    LAYER
                    TYPE LINE
                    STATUS ON
                    FEATURE
                      POINTS
                        751539 6669003
                        751539 6672326
                        755559 6672326
                        751539 6669003
                      END
                    END
                    CLASS
                      STYLE
                        COLOR 80 150 55
                        WIDTH 5
                      END
                    END
                  END
              ''')

    img = map.draw(
        bounds=gws.Bounds(
            extent=[738040, 6653804, 765743, 6683686],
            crs=crs.WEBMERCATOR,
        ),
        size=(800, 600),
    )

    image_original = image.from_path("/gws-app/gws/gis/ms/_test/vectors.png")

    assert img.compare_to(image_original) < 0.01


def test_geometry_style():
    u.pg.create('test_table',
                {'id': 'int primary key', 'geom': 'geometry(LINESTRING, 4326)', 'label': 'text'})

    data = [
        {'id': 4, 'geom': 'LINESTRING(0 -5,-10 0)', 'label': 'label 4'},
        {'id': 5, 'geom': 'LINESTRING(0 0, 5 0, 2.5 5, 0 0)', 'label': 'label 5'}
    ]

    u.pg.insert('test_table', data)

    map = ms.new_map()

    style_vals = gws.StyleValues(fill='blue',
                                 stroke='green',
                                 stroke_width=10,
                                 stroke_dasharray=[20, 40, -1],
                                 stroke_linejoin='miter',
                                 stroke_miterlimit=10,
                                 stroke_linecap='round',
                                 with_geometry='all',
                                 offset_x=10,
                                 offset_y=0,
                                 )

    vl_opts = ms.LayerOptions(type=ms.LayerType.line,
                              connectionType="postgres",
                              connectionString=u.pg.url(),
                              crs=crs.WGS84,
                              dataString="geom FROM test_table USING UNIQUE id USING SRID=4326",
                              style=style_vals)

    map.add_layer(vl_opts)

    img = map.draw(
        bounds=gws.Bounds(
            extent=[-15, -10, 15, 10],
            crs=crs.WGS84,
        ),
        size=(800, 600),
    )
    image_original = image.from_path("/gws-app/gws/gis/ms/_test/geometry_style.png")

    assert img.compare_to(image_original) < 0.01


def test_label_style():
    # setup and fill pg table with vectordata
    u.pg.create('test_table', {'id': 'int primary key', 'geom': 'geometry(POINT, 4326)', 'label': 'text'})

    data = [
        {'id': 1, 'geom': 'POINT(0 0)', 'label': 'label 1'},
        {'id': 2, 'geom': 'POINT(5 5)', 'label': 'label 2'}
    ]

    u.pg.insert('test_table', data)

    map = ms.new_map()

    style_vals = gws.StyleValues(fill='blue',
                                 stroke='green',
                                 stroke_width=10,
                                 with_geometry='all',

                                 with_label='all',
                                 label_align='left',
                                 label_background='red',
                                 label_fill='black',
                                 # label_font_family='sans',
                                 label_font_size=30,
                                 # label_font_style='italic',
                                 # label_font_weight='bold',
                                 # label_line_height=100,
                                 # label_min_scale=1000,
                                 # label_max_scale=10000,
                                 label_offset_x=10,
                                 label_offset_y=10,
                                 label_padding=[2, 3, 4, 5],
                                 label_placement='start',
                                 label_stroke='yellow',
                                 label_stroke_dasharray=[5, 10, -1],
                                 label_stroke_linecap='round',
                                 label_stroke_linejoin='miter',
                                 label_stroke_miterlimit=10,
                                 label_stroke_width=10,
                                 )

    vl_opts = ms.LayerOptions(type=ms.LayerType.point,
                              connectionType="postgres",
                              connectionString=u.pg.url(),
                              crs=crs.WGS84,
                              dataString="geom FROM test_table USING UNIQUE id USING SRID=4326",
                              style=style_vals)

    map.add_layer(vl_opts)

    img = map.draw(
        bounds=gws.Bounds(
            extent=[-15, -10, 15, 10],
            crs=crs.WGS84,
        ),
        size=(800, 600),
    )
    image_original = image.from_path("/gws-app/gws/gis/ms/_test/label_style.png")

    assert img.compare_to(image_original) < 0.01


def test_marker_style():
    u.pg.create('test_table', {'id': 'int primary key', 'geom': 'geometry(LINESTRING, 4326)'})

    data = [
        {'id': 1, 'geom': 'LINESTRING(-10 0, 10 0)'},
        {'id': 2, 'geom': 'LINESTRING(0 -5, 0 5)'}
    ]

    u.pg.insert('test_table', data)

    map = ms.new_map()

    style_vals = gws.StyleValues(fill='blue',
                                 stroke='green',
                                 stroke_width=10,
                                 with_geometry='none',

                                 marker='arrow',
                                 marker_fill='red',
                                 marker_stroke_dashoffset=-25,
                                 marker_stroke_linecap='round',
                                 marker_stroke_linejoin='miter',
                                 marker_stroke_miterlimit=10,
                                 marker_size=10
                                 )

    vl_opts = ms.LayerOptions(type=ms.LayerType.line,
                              connectionType="postgres",
                              connectionString=u.pg.url(),
                              crs=crs.WGS84,
                              dataString="geom FROM test_table USING UNIQUE id USING SRID=4326",
                              style=style_vals)

    map.add_layer(vl_opts)

    img = map.draw(
        bounds=gws.Bounds(
            extent=[-15, -10, 15, 10],
            crs=crs.WGS84,
        ),
        size=(800, 600),
    )
    image_original = image.from_path("/gws-app/gws/gis/ms/_test/marker_style.png")

    assert img.compare_to(image_original) < 0.01


def test_icon():
    create_cross()
    u.pg.create('test_table', {'id': 'int primary key', 'geom': 'geometry(POINT, 4326)', 'label': 'text'})

    data = [
        {'id': 1, 'geom': 'POINT(0 0)', 'label': 'label 1'},
        {'id': 2, 'geom': 'POINT(5 5)', 'label': 'label 2'}
    ]

    u.pg.insert('test_table', data)

    map = ms.new_map()

    style_vals = gws.StyleValues(fill='blue',
                                 stroke='green',
                                 stroke_width=10,
                                 with_geometry='all',
                                 icon='/data/cross.png',

                                 with_label='all',
                                 label_align='left',
                                 label_background='red',
                                 label_fill='black',
                                 # label_font_family='sans',
                                 label_font_size=30,
                                 # label_font_style='italic',
                                 # label_font_weight='bold',
                                 # label_line_height=100,
                                 # label_min_scale=1000,
                                 # label_max_scale=10000,
                                 label_offset_x=10,
                                 label_offset_y=10,
                                 label_padding=[2, 3, 4, 5],
                                 label_placement='start',
                                 label_stroke='yellow',
                                 label_stroke_dasharray=[5, 10, -1],
                                 label_stroke_linecap='round',
                                 label_stroke_linejoin='miter',
                                 label_stroke_miterlimit=10,
                                 label_stroke_width=10,
                                 )

    vl_opts = ms.LayerOptions(type=ms.LayerType.point,
                              connectionType="postgres",
                              connectionString=u.pg.url(),
                              crs=crs.WGS84,
                              dataString="geom FROM test_table USING UNIQUE id USING SRID=4326",
                              style=style_vals)

    map.add_layer(vl_opts)

    img = map.draw(
        bounds=gws.Bounds(
            extent=[-15, -10, 15, 10],
            crs=crs.WGS84,
        ),
        size=(800, 600),
    )
    image_original = image.from_path("/gws-app/gws/gis/ms/_test/icon.png")

    assert img.compare_to(image_original) < 0.01
