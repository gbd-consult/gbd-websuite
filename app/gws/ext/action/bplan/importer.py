import re
import shutil
import zipfile

import gws
import gws.ext.db.provider.postgres
import gws.gis.extent
import gws.gis.gdal2
import gws.gis.shape
import gws.qgis.project
import gws.tools.json2
import gws.tools.os2 as os2
import gws.tools.job

import gws.types as t


class Stats(t.Data):
    numRecords: int
    numPngs: int
    numPdfs: int


def run(action, src_path: str, replace: bool, au_uid: str = None, job: gws.tools.job.Job = None) -> Stats:
    """"Import bplan data from a file or a directory."""

    tmp_dir = None

    if os2.is_file(src_path):
        # a file is given - unpack it into a temp dir
        tmp_dir = gws.ensure_dir(gws.TMP_DIR + '/bplan_' + gws.random_string(32))
        _extract(src_path, tmp_dir)

    stats = None

    try:
        stats = _run2(action, tmp_dir or src_path, replace, au_uid, job)
    except gws.tools.job.PrematureTermination as e:
        pass

    if tmp_dir:
        shutil.rmtree(tmp_dir)

    return stats


def update(action):
    with action.db.connect() as conn:
        rs = conn.select(f'SELECT DISTINCT _au FROM {conn.quote_table(action.plan_table.name)}')
        au_uids = set(r['_au'] for r in rs)

    _create_vrts(action, au_uids)
    _update_pdfs(action, au_uids)
    _create_qgis_projects(action, au_uids)


##

def _run2(action, src_dir, replace, au_uid, job):
    gws.log.debug(f'BEGIN {src_dir!r} au={au_uid!r}')

    stats = Stats(numRecords=0, numPngs=0, numPdfs=0)

    _update_job(job, step=0, steps=6)

    # iterate shape files and prepare a list of db records

    shp_paths = set()
    recs = {}

    for p in sorted(os2.find_files(src_dir, ext='shp')):
        # NB prefer '..._utf8.shp' variants if they exist
        if 'utf8' in p:
            shp_paths.discard(p.replace('utf8', ''))
        shp_paths.add(p)

    for p in sorted(shp_paths):
        gws.log.debug(f'read {p!r}')

        if au_uid and not _path_belongs_to_au(p, [au_uid]):
            continue

        with gws.gis.gdal2.from_path(p) as ds:
            for f in gws.gis.gdal2.features(ds, action.crs, encoding=_encoding(p)):
                r = {a.name.lower(): a.value for a in f.attributes}

                r['_uid'] = uid = r[action.key_col]
                r['_au'] = r[action.au_key_col]
                r['_type'] = action.type_mapping.get(r.get(action.type_col, ''), '')

                if uid not in recs:
                    recs[uid] = r

                if not f.shape:
                    # if no geometry found, create a point from x/y coords
                    try:
                        f.shape = gws.gis.shape.from_geometry({
                            "type": "Point",
                            "coordinates": [
                                float(r[action.x_coord_col]),
                                float(r[action.y_coord_col]),
                            ]
                        }, action.crs)
                    except:
                        pass

                if f.shape:
                    s = f.shape.to_multi()
                    recs[uid][_geom_name(s)] = s.ewkt

    _update_job(job, step=1)

    # insert records

    table: t.SqlTable = action.plan_table
    db: gws.ext.db.provider.postgres.Object = action.db

    recs = list(recs.values())

    au_uids = [au_uid] if au_uid else sorted(set(r['_au'] for r in recs))

    with db.connect() as conn:
        src = table.name

        with conn.transaction():
            for a in au_uids:
                au_recs = [r for r in recs if r['_au'] == a]
                if not au_recs:
                    continue

                gws.log.debug(f'insert {a!r} ({len(au_recs)})')
                stats.numRecords += len(au_recs)

                if replace:
                    conn.execute(f'DELETE FROM {conn.quote_table(src)} WHERE _au = %s', [a])
                else:
                    uids = [r['_uid'] for r in au_recs]
                    ph = ','.join(['%s'] * len(uids))
                    conn.execute(f'DELETE FROM {conn.quote_table(src)} WHERE _uid IN ({ph})', uids)

                conn.insert_many(src, au_recs)

    _update_job(job, step=2)

    # move png/pgw files into place

    dd = action.data_dir

    if replace:
        for p in os2.find_files(f'{dd}/png'):
            if _path_belongs_to_au(p, au_uids):
                gws.log.debug(f'delete {p}')
                os2.unlink(p)

    for p in os2.find_files(src_dir, ext='png'):
        if not _path_belongs_to_au(p, au_uids):
            continue

        w = re.sub(r'\.png$', '.pgw', p)
        if not os2.is_file(w):
            continue

        fb = _fnbody(p)

        gws.log.debug(f'copy {fb}.png')
        shutil.copyfile(p, f'{dd}/png/{fb}.png')
        shutil.copyfile(w, f'{dd}/png/{fb}.pgw')

        stats.numPngs += 1

    _update_job(job, step=3)

    # move pdfs into place

    if replace:
        for p in os2.find_files(f'{dd}/pdf'):
            if _path_belongs_to_au(p, au_uids):
                gws.log.debug(f'delete {p}')
                os2.unlink(p)

    for p in os2.find_files(src_dir, ext='pdf'):
        if not _path_belongs_to_au(p, au_uids):
            continue

        fb = _fnbody(p)

        gws.log.debug(f'copy {fb}.pdf')
        shutil.copyfile(p, f'{dd}/pdf/{fb}.pdf')

        stats.numPdfs += 1

    _update_job(job, step=4)

    #

    _create_vrts(action, au_uids)
    _update_job(job, step=5)

    #

    _update_pdfs(action, au_uids)
    _update_job(job, step=6)

    #

    _create_qgis_projects(action, au_uids)
    _update_job(job, state=gws.tools.job.State.complete)

    gws.log.debug(f'END {src_dir!r}')

    return stats


def _create_vrts(action, au_uids):
    """Create VRT files from png/pgw files, one VRT per au + typecode."""

    gws.log.debug(f'create vrts for {au_uids!r}')

    dd = action.data_dir
    pngs = [_filename(p) for p in os2.find_files(dd + '/png', ext='png')]

    groups = {}

    with action.db.connect() as conn:
        for r in conn.select(f'SELECT _uid, _au, _type FROM {conn.quote_table(action.plan_table.name)}'):
            if r['_au'] in au_uids:
                layer_uid = _qgis_layer_uid(r, geom_type='r')
                groups.setdefault(layer_uid, []).extend(p for p in pngs if p.startswith(r['_uid']))

    for layer_uid, pngs in sorted(groups.items()):
        vrt = f'{dd}/vrt/{layer_uid}.vrt'

        os2.unlink(vrt)

        if not pngs:
            continue

        gws.log.debug(f'create {layer_uid}.vrt')

        lst = f'{dd}/vrt/{layer_uid}.lst'
        gws.write_file(lst, '\n'.join(f'{dd}/png/{p}' for p in pngs))
        os2.run([
            'gdalbuildvrt',
            '-srcnodata', '0',
            '-input_file_list', lst,
            '-overwrite', vrt
        ])


def _update_pdfs(action, au_uids):
    gws.log.debug(f'update pdfs for {au_uids!r}')

    dd = action.data_dir
    by_uid = {}

    with action.db.connect() as conn:
        for r in conn.select(f'SELECT _au, _uid FROM {conn.quote_table(action.plan_table.name)}'):
            if r['_au'] in au_uids:
                by_uid[r['_uid']] = []

    for p in os2.find_files(dd + '/pdf', ext='pdf'):
        fn = _filename(p)
        for uid, names in by_uid.items():
            if fn.startswith(uid):
                names.append(fn)
                break

    with action.db.connect() as conn:
        with conn.transaction():
            for uid, names in by_uid.items():
                if names:
                    gws.log.debug(f'save pdfs for {uid}')
                    names = ','.join(names)
                    conn.execute(f'UPDATE {conn.quote_table(action.plan_table.name)} SET _pdf=%s WHERE _uid=%s', [names, uid])


def _create_qgis_projects(action, au_uids):
    gws.log.debug(f'create qgis projects for {au_uids!r}')

    dd = action.data_dir
    layer_uids = set()
    extents = {}
    template = gws.read_file(action.qgis_template)

    with action.db.connect() as conn:

        tab = conn.quote_table(action.plan_table.name)

        rs = conn.select(f'SELECT _au, ST_Extent(_geom_p) AS p FROM {tab} GROUP BY _au')
        for r in rs:
            if r['_au'] in au_uids:
                extents[r['_au']] = gws.gis.extent.from_box(r['p'])

        for g in 'plx':
            rs = conn.select(f'SELECT DISTINCT _au, _type FROM {tab} WHERE _geom_{g} IS NOT NULL')
            for r in rs:
                if r['_au'] in au_uids:
                    layer_uids.add(_qgis_layer_uid(r, geom_type=g))

    for p in os2.find_files(dd + '/vrt', ext='vrt'):
        layer_uids.add(_fnbody(p))

    for au in action.au_list:
        ext = extents.get(au.uid)
        if not ext:
            continue

        xml = template
        xml = xml.replace('{au.uid}', au.uid)
        xml = xml.replace('{au.name}', au.name)

        prj = gws.qgis.project.from_string(xml)

        for e in prj.bs.select('extent xmin'):
            e.string = str(ext[0])
        for e in prj.bs.select('extent ymin'):
            e.string = str(ext[1])
        for e in prj.bs.select('extent xmax'):
            e.string = str(ext[2])
        for e in prj.bs.select('extent ymax'):
            e.string = str(ext[3])

        for e in prj.bs.select('layer-tree-layer'):
            if e['id'] not in layer_uids:
                e.decompose()

        for e in prj.bs.select('maplayer'):
            if e.id.text not in layer_uids:
                e.decompose()

        path = f'{dd}/qgs/{au.uid}.qgs'
        gws.log.debug(f'create {path}')
        prj.save(path)


def _qgis_layer_uid(rec, geom_type):
    return rec['_type'].lower() + '_' + geom_type + '_' + rec['_au']


def _extract(zip_path, target_dir):
    zf = zipfile.ZipFile(zip_path)
    for fi in zf.infolist():
        fn = _filename(fi.filename)
        if not fn or fn.startswith('.'):
            continue
        with zf.open(fi) as src, open(target_dir + '/' + fn, 'wb') as dst:
            gws.log.debug(f'unzip {fn!r}')
            shutil.copyfileobj(src, dst)


def _encoding(path):
    if os2.is_file(path.replace('.shp', '.cpg')):
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


def _fnbody(path):
    return os2.parse_path(path)['name']


def _path_belongs_to_au(path, au_uids):
    fn = _filename(path)
    # filename is like AAAAnnn.png or Shapes_AAAA_xxx.shp, where AAAA = au uid
    for aid in au_uids:
        if fn.startswith(aid) or (aid + '_' in fn) or ('_' + aid in fn):
            return True
    return False


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
