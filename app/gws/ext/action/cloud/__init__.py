import re
from lxml import etree

import gws
import gws.config
import gws.config.parser
import gws.ext.db.provider.postgres
import gws.server
import gws.web.error
import gws.tools.mime
import gws.tools.job
import gws.tools.json2
import gws.tools.misc
import gws.gis.extent

import gws.types as t
import gws.common.template


class Config(t.WithTypeAndAccess):
    """Cloud admin action"""
    pass


class DataSet(t.Data):
    """Data set for a cloud project."""
    uid: str  #: data set uid as used in the map
    records: t.List[dict]  #: list of data records


class AssetType(t.Enum):
    svg = "svg"


class Source(t.Data):
    text: t.Optional[str]  #: text content
    content: t.Optional[bytes]  #: binary content


class Asset(t.Data):
    """Media asset for the map"""

    type: AssetType  #: type, e.g. "svg"
    name: str  #: file name, as used in the map source
    source: Source  #: file source


class MapType(t.Enum):
    qgis = "qgis"


class Map(t.Data):
    """Cloud project map"""

    type: MapType  #: map type
    source: Source  #: map source code (e.g. a QGIS project)
    layerUids: t.Optional[t.List[str]]  #: layer ids to use in the cloud
    data: t.Optional[t.List[DataSet]]  #: data for the map
    assets: t.Optional[t.List[Asset]]  #: map assets
    extent: t.Optional[t.Extent]
    scales: t.Optional[t.List[int]]
    initScale: t.Optional[int]
    center: t.Optional[t.Point]


class CreateProjectParams(t.Params):
    """Parameters for the CreateProject command"""

    userUid: str  #: API user ID
    userKey: str  #: API user key
    projectName: str  #: project name
    map: Map  #: project map


class CreateProjectResponse(t.Response):
    pass


_QGIS_DATASOURCE_TEMPLATE = """dbname='{database}' host={host} port={port} user='{user}' password='{password}' sslmode=disable key='{key}' srid={geo_srid} type={geo_type} table="{schema}"."{table}" ({geo_col}) sql="""

_CONFIG_TEMPLATE_PATH = '/data/cloud_project_template.json'

_DEFAULT_SCALES = [
    4000000, 2000000, 1000000, 500000, 250000, 150000, 70000, 35000, 15000, 8000, 4000, 2000, 1000
]

CLOUD_DIR = gws.VAR_DIR + '/cloud'
CLOUD_USER_DIR = CLOUD_DIR + '/users'
CLOUD_CONFIG_DIR = CLOUD_DIR + '/configs'


class Object(gws.ActionObject):
    def __init__(self):
        super().__init__()
        self.db: gws.ext.db.provider.postgres = None

    def configure(self):
        super().configure()
        p = self.var('db')
        self.db = self.root.find('gws.ext.db.provider', p) if p else self.root.find_first(
            'gws.ext.db.provider.postgres')

    def api_create_project(self, req: t.IRequest, p: CreateProjectParams) -> CreateProjectResponse:
        # debug
        with open(gws.VAR_DIR + '/cloud-debug-input.json', 'w') as fp:
            fp.write(gws.tools.json2.to_string(p, pretty=True))
        with open(gws.VAR_DIR + '/cloud-debug-input.qgs', 'w') as fp:
            fp.write(p.map.source.text)

        uid = self._create_project(p)

        return CreateProjectResponse(
            url=req.url_for(gws.SERVER_ENDPOINT + f'/cmd/assetHttpGetPath/projectUid/{uid}/path/project.cx.html')
        )

    def _create_project(self, p: CreateProjectParams):
        user_uid = gws.as_uid(p.userUid)
        project_uid = gws.as_uid(p.projectName)

        user_dir = gws.tools.misc.ensure_dir(f'{CLOUD_USER_DIR}/{user_uid}')

        with self.db.connect() as conn:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS {user_uid}')

        ds_map = {}

        for ds in p.map.data:
            ds_map[ds.uid] = self._dataset_to_table(ds, user_uid)

        assets_dir = gws.tools.misc.ensure_dir(f'{user_dir}/assets')
        qdata = self._prepare_qgis_project(p.map.source.text, ds_map, p.map.assets, assets_dir)
        # use a custom ext to prevent the monitor from watching this
        # @TODO fix monitor settings
        qpath = f'{user_dir}/{project_uid}.qgs.x'

        with open(qpath, 'w') as fp:
            fp.write(qdata['source'])

        project_full_uid = user_uid + '::' + project_uid

        tpl_vars = {
            'PROJECT_UID': project_full_uid,
            'PROJECT_TITLE': p.projectName,
            'MAP_EXTENT': p.map.extent or qdata['extent'],
            'MAP_SCALES': p.map.scales or _DEFAULT_SCALES,
            'MAP_CRS': qdata['crs'],
            'QGIS_PATH': qpath,
            'QGIS_LAYERS': p.map.layerUids,
        }

        tpl_vars['MAP_CENTER'] = p.map.center or gws.gis.extent.center(tpl_vars['MAP_EXTENT'])
        tpl_vars['MAP_INIT_SCALE'] = p.map.initScale or tpl_vars['MAP_SCALES'][0]

        with open(_CONFIG_TEMPLATE_PATH) as fp:
            tpl = fp.read()

        config = re.sub(r'"{{(\w+)}}"', lambda m: gws.tools.json2.to_string(tpl_vars[m.group(1)]), tpl)
        config = gws.tools.json2.from_string(config)

        with open(gws.VAR_DIR + '/cloud-debug-config.json', 'w') as fp:
            fp.write(gws.tools.json2.to_string(config, pretty=True))

        # parse the config before saving, if this fails, then it fails
        gws.log.debug('parsing config...')
        gws.config.parser.parse(config, 'gws.common.project.Config')
        gws.log.debug('parsing config ok')

        config_dir = gws.tools.misc.ensure_dir(CLOUD_CONFIG_DIR)
        cfg_path = config_dir + f'/{project_full_uid}.config.json'
        gws.tools.json2.to_path(cfg_path, config, pretty=True)

        gws.log.debug(f'added project {project_full_uid!r}')

        return project_full_uid

    def _dataset_to_table(self, ds: DataSet, schema_name):
        ddl = []

        qsrc = dict(self.db.connect_params)
        qsrc['schema'] = schema_name
        qsrc['table'] = gws.as_uid(ds.uid)

        # @TODO require the client to provide a data schema

        for k, v in ds.records[0].items():
            pt = 'TEXT'

            if isinstance(v, int) and k == 'id':
                pt = 'INT PRIMARY KEY'
                qsrc['key'] = k
            elif isinstance(v, (int, bool)):
                pt = 'INT'
            elif isinstance(v, float):
                pt = 'FLOAT'
            elif isinstance(v, dict) and 'geometry' in v:
                qsrc['geo_col'] = k
                qsrc['geo_srid'] = v['crs'].split(':')[1]
                qsrc['geo_type'] = v['geometry']['type']
                pt = 'GEOMETRY(%s,%s)' % (qsrc['geo_type'], qsrc['geo_srid'])
                # insert geometries as json first and have postgis convert them
                ddl.append(k + '__json TEXT')

            ddl.append(k + ' ' + pt)

        table = qsrc['schema'] + '.' + qsrc['table']

        ddl = ',\n'.join(ddl)
        create_sql = f'CREATE TABLE {table} ({ddl})'

        data = []
        geo_col = qsrc.get('geo_col')

        for rec in ds.records:
            r = {}
            for k, v in rec.items():
                if k == geo_col:
                    g = v['geometry']
                    g['crs'] = {'type': 'name', 'properties': {'name': v['crs']}}
                    r[k + '__json'] = gws.tools.json2.to_string(g)
                else:
                    r[k] = v
            data.append(r)

        with self.db.connect() as conn:
            conn.execute(f'DROP TABLE IF EXISTS {table}')
            conn.execute(create_sql)
            conn.insert_many(table, data)
            if geo_col:
                conn.execute(f'UPDATE {table} SET {geo_col}=ST_GeomFromGeoJSON({geo_col}__json)')
                conn.execute(f'ALTER TABLE {table} DROP COLUMN {geo_col}__json')

        gws.log.debug(f'added tabled {table!r} qsrc={qsrc!r}')

        return qsrc

    def _prepare_qgis_project(self, text, ds_map, assets, assets_dir):
        # @TODO move this to qqgis

        source_strings = {
            uid: gws.tools.misc.format_placeholders(_QGIS_DATASOURCE_TEMPLATE, qsrc)
            for uid, qsrc in ds_map.items()
        }

        xml = etree.fromstring(text)

        for node in xml.findall('.//layer-tree-layer'):
            uid = node.get('id')
            if uid in source_strings:
                gws.p('replace source', node.get('source'), source_strings[uid])
                node.set('source', source_strings[uid])
                node.set('providerKey', 'postgres')

        for node in xml.findall('.//maplayer'):
            n = node.find('id')
            if n is not None and n.text in ds_map:
                uid = n.text
                n = node.find('datasource')
                if n is not None:
                    gws.p('replace source', n.text, source_strings[uid])
                    n.text = source_strings[uid]
                n = node.find('provider')
                if n is not None:
                    n.text = 'postgres'

        if not xml.findall('.//WMSUseLayerIDs'):
            etree.SubElement(xml.find('properties'), 'WMSUseLayerIDs')

        for node in xml.findall('.//WMSUseLayerIDs'):
            node.text = 'true'

        source = etree.tounicode(xml)

        for a in assets:
            path = assets_dir + '/' + gws.as_uid(a.name.split('.')[0]) + '.' + a.type
            if a.source.text:
                with open(path, 'w') as fp:
                    fp.write(a.source.text)
            if a.source.content:
                with open(path, 'wb') as fp:
                    fp.write(gws.as_bytes(a.source.content))
            source = source.replace(f'"{a.name}"', f'"{path}"')
            gws.p(f'saved asset {a.name!r} to {path!r}')

        qdata = {
            'crs': '',
            'extent': [],
            'source': source,
        }

        for node in xml.findall('.//WMSExtent/value'):
            qdata['extent'].append(float(node.text))

        for node in xml.findall('.//projectCrs/spatialrefsys/authid'):
            qdata['crs'] = node.text

        return qdata
