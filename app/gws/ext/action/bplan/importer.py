import re
import shutil
import zipfile

import gws
import gws.ext.db.provider.postgres
import gws.gis.extent
import gws.gis.gdal2
import gws.qgis.project
import gws.tools.json2
import gws.tools.os2 as os2
import gws.tools.job

import gws.types as t


def run(action, src_path: str, replace: bool, job: gws.tools.job.Job = None):
    """"Import bplan data from a file or a directory."""

    tmp_dir = None

    if os2.is_file(src_path):
        # a file is given - unpack it into a temp dir
        tmp_dir = gws.ensure_dir(gws.TMP_DIR + '/bplan_' + gws.random_string(32))
        _extract(src_path, tmp_dir)

    try:
        _run2(action, tmp_dir or src_path, replace, job)
    except gws.tools.job.PrematureTermination as e:
        pass

    # @TODO remove tmp_dir


def update(action):
    _create_vrts(action)
    _update_pdfs(action)
    _create_qgis_projects(action)


##

def _run2(action, src_dir, replace, job):
    gws.log.debug(f'BEGIN {src_dir!r}')

    _update_job(job, step=0, steps=6)

    # iterate shape files and prepare a list of db records

    shp_paths = set()
    recs = {}

    for p in sorted(os2.find_files(src_dir, ext='shp')):
        # NB prefer '..._utf8.shp' variants if they exist
        if '_utf8' in p:
            shp_paths.discard(p.replace('_utf8', ''))
        shp_paths.add(p)

    for p in shp_paths:
        gws.log.debug(f'read {p!r}')
        with gws.gis.gdal2.from_path(p) as ds:
            for f in gws.gis.gdal2.features(ds, action.crs, encoding=_encoding(p)):
                r = {a.name.lower(): a.value for a in f.attributes}

                r['_uid'] = uid = r[action.key_col]
                r['_type'] = action.type_mapping.get(r.get(action.type_col, ''), '')

                if uid not in recs:
                    recs[uid] = r
                if f.shape:
                    s = f.shape.to_multi()
                    recs[uid][_geom_name(s)] = s.ewkt

    _update_job(job, step=1)

    # insert records

    table: t.SqlTable = action.table
    db: gws.ext.db.provider.postgres.Object = action.db

    aukey = action.au_key_col
    recs = list(recs.values())
    aus = set(r.get(aukey) for r in recs)

    with db.connect() as conn:
        src = table.name
        conn.execute(f'TRUNCATE {conn.quote_table(src)}')


        with conn.transaction():
            for au in sorted(aus):
                au_recs = [r for r in recs if r.get(aukey) == au]
                gws.log.debug(f'insert {au!r} ({len(au_recs)})')

                if replace:
                    conn.execute(f'DELETE FROM {conn.quote_table(src)} WHERE {aukey} = %s', [au])
                else:
                    uids = [r['_uid'] for r in au_recs]
                    ph = ','.join(['%s'] * len(uids))
                    conn.execute(f'DELETE FROM {conn.quote_table(src)} WHERE _uid IN ({ph})', uids)

                conn.insert_many(src, au_recs)

    _update_job(job, step=2)

    # move png/pgw files into place

    dd = action.data_dir

    # if replace:
    #     for au in aus:
    #         for p in os2.find_files(raster_dir, '\.png$'):
    #             if p.startswith(au):
    #                 print('delete ', p)

    for p in os2.find_files(src_dir, ext='png'):
        cc = _filecode(p)
        if not cc:
            continue

        w = re.sub(r'\.png$', '.pgw', p)
        if not os2.is_file(w):
            continue

        gws.log.debug(f'copy {cc}.png')
        shutil.copyfile(p, f'{dd}/png/{cc}.png')
        shutil.copyfile(w, f'{dd}/png/{cc}.pgw')

    _update_job(job, step=3)

    # move pdfs into place

    for p in os2.find_files(src_dir, ext='pdf'):
        cc = _filecode(p)
        if not cc:
            continue

        gws.log.debug(f'copy {cc}.pdf')
        shutil.copyfile(p, f'{dd}/pdf/{cc}.pdf')

    _update_job(job, step=4)

    #

    _create_vrts(action)
    _update_job(job, step=5)

    #

    _update_pdfs(action)
    _update_job(job, step=6)

    #

    _create_qgis_projects(action)
    _update_job(job, state=gws.tools.job.State.complete)

    gws.log.debug(f'END {src_dir!r}')


_EMPTY_VRT = '<VRTDataset rasterXSize="0" rasterYSize="0"/>'


def _create_vrts(action):
    """Create VRT files from png/pgw files, one VRT per au + typecode."""

    akey = action.au_key_col
    dd = action.data_dir
    pngs = [_filename(p) for p in os2.find_files(dd + '/png', ext='png')]

    groups = {}

    with action.db.connect() as conn:
        for r in conn.select(f'SELECT _uid, {akey}, _type FROM {conn.quote_table(action.table.name)}'):
            key = r[akey]
            if r['_type']:
                key += '-' + r['_type']
            groups.setdefault(key, []).extend(p for p in pngs if p.startswith(r['_uid']))

    for key, ps in sorted(groups.items()):
        vrt = f'{dd}/vrt/{key}.vrt'

        if not ps:
            gws.write_file(vrt, _EMPTY_VRT)
            continue

        gws.log.debug(f'create {key}.vrt')

        lst = f'{dd}/vrt/{key}.lst'
        gws.write_file(lst, '\n'.join(f'{dd}/png/{p}' for p in ps))
        os2.run([
            'gdalbuildvrt',
            '-srcnodata', '0',
            '-input_file_list', lst,
            '-overwrite', vrt
        ])


def _update_pdfs(action):
    gws.log.debug(f'update pdf lists')

    dd = action.data_dir
    by_uid = {}

    with action.db.connect() as conn:
        for r in conn.select(f'SELECT _uid FROM {conn.quote_table(action.table.name)}'):
            by_uid[r['_uid']] = []

    for p in os2.find_files(dd + '/pdf', ext='pdf'):
        p = _filename(p)
        for uid, names in by_uid.items():
            if p.startswith(uid):
                names.append(p)
                break

    with action.db.connect() as conn:
        with conn.transaction():
            for uid, names in by_uid.items():
                if names:
                    gws.log.debug(f'save pdfs for {uid}')
                    names = ','.join(names)
                    conn.execute(f'UPDATE {conn.quote_table(action.table.name)} SET _pdf=%s WHERE _uid=%s', [names, uid])


def _create_qgis_projects(action):
    akey = action.au_key_col
    dd = action.data_dir

    xml = gws.read_file(action.qgis_template)
    m = re.search(r'(\w+)\.vrt', xml)
    placeholder = m.group(1)

    extent_tags = 'xmin', 'ymin', 'xmax', 'ymax'

    with action.db.connect() as conn:
        rs = conn.select(f'''
            SELECT {akey} AS a, ST_Extent(_geom_p) AS e
            FROM {conn.quote_table(action.table.name)}
            GROUP BY {akey}
        ''')
        for r in rs:
            au = r['a']
            ext = gws.gis.extent.from_box(r['e'])

            gws.log.debug(f'create {au}.qgs')

            prj = gws.qgis.project.from_string(xml.replace(placeholder, au))
            for n, val in enumerate(ext):
                for e in prj.bs.select('extent ' + extent_tags[n]):
                    e.string = str(val)

            prj.save(f'{dd}/qgs/{au}.qgs')


def _extract(zip_path, target_dir):
    zf = zipfile.ZipFile(zip_path)
    for fi in zf.infolist():
        fn = re.sub(r'[^\w.-]', '_', fi.filename)
        if fn.startswith('.'):
            continue
        with zf.open(fi) as src, open(target_dir + '/' + fn, 'wb') as dst:
            gws.log.debug(f'unzip {fn!r}')
            shutil.copyfileobj(src, dst)


def _encoding(path):
    p = path.replace('.shp', '.cpg')
    if os2.is_file(p):
        # have a cpg file, let gdal handle the encoding
        return
    return 'utf8' if 'utf8' in path else 'ISO-8859–1'


def _geom_name(s: t.IShape):
    if s.type == t.GeometryType.multipoint:
        return '_geom_x'
    if s.type == t.GeometryType.multilinestring:
        return '_geom_l'
    if s.type == t.GeometryType.multipolygon:
        return '_geom_p'
    raise ValueError(f'invalid geometry type: {s.type!r}')


def _filename(path):
    return os2.parse_path(path)['filename']


def _filecode(path):
    m = re.search(r'\b([0-9A-Z]+)\.[a-z]+$', path)
    return m.group(1) if m else None


def _diff(a, b):
    d = {}
    for k in a.keys() | b.keys():
        if a.get(k) != b.get(k):
            d[k] = a[k], b[k]
    return d


def _update_job(job, **kwargs):
    if not job:
        return

    j = gws.tools.job.get(job.uid)

    if not j:
        raise gws.tools.job.PrematureTermination('NOT_FOUND')

    if j.state != gws.tools.job.State.running:
        raise gws.tools.job.PrematureTermination(f'WRONG_STATE={j.state}')

    j.update(**kwargs)
