"""Shared fixtures for qfieldcloud tests."""

import os

import gws
import gws.lib.xmlx
import gws.test.util as u

SCHEMA = 'qfc'

SOURCE_QGS = os.path.dirname(__file__) + '/qfield.qgs'

# the qgis container mounts VAR_DIR under the same path, so projects placed here
# are readable by the qgis server as well
WORK_DIR = f'{gws.c.VAR_DIR}/qfieldcloud_test'


def qgs_path(name: str = 'qfield', patch=None) -> str:
    """Copy the fixture qgis project into the shared work dir and return its path.

    Args:
        name: Base name for the copy.
        patch: Optional callable, receives the parsed xml root before writing.
    """

    text = gws.u.read_file(SOURCE_QGS)
    text = text.replace('{PG_SERVICE}', u.option('service.postgres.name'))

    if patch:
        root_el = gws.lib.xmlx.from_string(text)
        patch(root_el)
        text = root_el.to_string()

    path = f'{gws.u.ensure_dir(WORK_DIR)}/{name}.qgs'
    gws.u.write_file(path, text)
    return path


def set_project_prop(root_el: gws.XmlElement, name: str, value: str, type: str = 'QString'):
    """Set a qfieldsync project property, for use as a `qgs_path` patch."""

    el = root_el.require('properties/qfieldsync')
    old = el.find(name)
    if old is not None:
        el.remove(old)
    el.add(name, {'type': type}).text = value


def remove_media_dirs(root_el: gws.XmlElement):
    """Drop the media dirs, for use as a `qgs_path` patch.

    All fixture projects live in the same directory, so a test that creates media files
    would otherwise affect every other project as well.
    """

    el = root_el.require('properties')
    old = el.find('QFieldSync')
    if old is not None:
        el.remove(old)


def set_layer_prop(root_el: gws.XmlElement, layer_id: str, name: str, value: str, type: str = 'QString'):
    """Set a QFieldSync layer custom property, for use as a `qgs_path` patch."""

    for el in root_el.findall('projectlayers/maplayer'):
        if el.textof('id') != layer_id:
            continue
        opt = el.require('customproperties/Option')
        for o in opt.findall('Option'):
            if o.get('name') == f'QFieldSync/{name}':
                opt.remove(o)
        opt.add('Option', {'name': f'QFieldSync/{name}', 'type': type, 'value': value})
        return
    raise ValueError(f'layer {layer_id!r} not found')


def create_tables():
    """Create the postgres tables the fixture project refers to."""

    u.pg.create_schema(SCHEMA)

    u.pg.create(f'{SCHEMA}.poi', {
        'id': 'int primary key',
        'name': 'text',
        'district_id': 'int',
        'photo': 'text',
        'photo_content': 'bytea',
        'geom': 'geometry(Point,3857)',
    })

    u.pg.create(f'{SCHEMA}.district', {
        'id': 'int primary key',
        'name': 'text',
        'geom': 'geometry(MultiPolygon,3857)',
    })

    u.pg.create(f'{SCHEMA}.note', {
        'id': 'serial primary key',
        'fid': 'int',
        'kind': 'text',
        'text': 'text',
        'geom': 'geometry(Point,3857)',
    })


def point(x, y):
    return u.pg.ewkb(f'POINT({x} {y})', 3857)


def polygon(x, y, size=1000):
    return u.pg.ewkb(
        f'MULTIPOLYGON((({x} {y}, {x + size} {y}, {x + size} {y + size}, {x} {y + size}, {x} {y})))',
        3857,
    )
