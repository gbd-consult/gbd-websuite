import re
import shutil
import zipfile
import PIL.Image

import gws
import gws.ext.db.provider.postgres
import gws.gis.extent
import gws.gis.gdal2
import gws.gis.shape
import gws.qgis.project
import gws.tools.json2
import gws.tools.date
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

    _update_pdfs(action, au_uids)
    _create_qgis_projects(action, au_uids)


def delete_feature(action, uid):
    if not uid:
        return

    with action.db.connect() as conn:
        conn.execute(f'''
            DELETE
            FROM {conn.quote_table(action.plan_table.name)}
            WHERE _uid = %s
        ''', [uid])

    dd = action.data_dir

    _delete_feature_assets(dd + '/pdf', uid)
    _delete_feature_assets(dd + '/png', uid)
    _delete_feature_assets(dd + '/cnv', uid)


##

_DATE_FIELDS = [
    'AKT_DATENB',
    'AKT_DATENS',
    'AKT_RECHT',
    'AUFSTELLB',
    'EINLEITB',
    'FESTSTELLB',
    'OFFENLEGB',
    'RECHTSKR',
    'SATZBESCHL',
]


def _to_date_str(val):
    if not val:
        return ''

    val = str(val).strip()
    if not val:
        return ''

    m = re.match(r'^(\d+)-(\d+)-(\d+)$', val)
    if m:
        return '%04d-%02d-%02d' % (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.match(r'^(\d+)\.(\d+).(\d+)$', val)
    if m:
        return '%04d-%02d-%02d' % (int(m.group(3)), int(m.group(2)), int(m.group(1)))

    gws.log.warn(f'invalid date: {val!r}')
    return ''


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
                r = {}

                # convert all attributes to strings
                for a in f.attributes:
                    if a.type == t.AttributeType.datetime:
                        val = gws.tools.date.to_iso_date(a.value)
                    elif a.name in _DATE_FIELDS:
                        val = _to_date_str(a.value)
                    else:
                        val = str(a.value)
                    r[a.name.lower()] = val

                r['_uid'] = uid = r[action.key_col]
                r['_au'] = r[action.au_key_col]

                type_name = r.get(action.type_col, '')
                for ty in action.type_list:
                    if ty.srcName == type_name:
                        r['_type'] = ty.uid
                        break

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
        os2.chown(f'{dd}/pdf/{fb}.pdf')

        stats.numPdfs += 1

    _update_job(job, step=4)

    #

    _update_pdfs(action, au_uids)
    _update_job(job, step=5)

    #

    _create_qgis_projects(action, au_uids)
    _update_job(job, state=gws.tools.job.State.complete)

    gws.log.debug(f'END {src_dir!r}')

    return stats


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
                    conn.execute(f'UPDATE {conn.quote_table(action.plan_table.name)} SET medien=%s WHERE _uid=%s', [names, uid])


def _create_qgis_projects(action, au_uids):
    gws.log.debug(f'create qgis projects for {au_uids!r}')

    dd = action.data_dir

    extents = _enum_extents(action, au_uids)
    layers = _enum_layers(action, au_uids)

    for au_uid in au_uids:
        path = f'{dd}/qgs/{au_uid}.qgs'

        ls = [la for la in layers if la['au_uid'] == au_uid]
        if not ls:
            os2.unlink(path)
            continue

        ext = extents.get(au_uid)
        if not ext:
            continue

        au_props = [au for au in action.au_list if au.uid == au_uid]
        if not au_props:
            continue

        res = action.qgis_template.render({
            'extent': ext,
            'layers': ls,
            'au': au_props[0],
        })

        gws.write_file(path, res.content)
        gws.log.debug(f'created {path!r}')


def _enum_extents(action, au_uids):
    extents = {}

    with action.db.connect() as conn:
        tab = conn.quote_table(action.plan_table.name)
        rs = conn.select(f'SELECT _au, ST_Extent(_geom_p) AS p FROM {tab} GROUP BY _au')
        for rec in rs:
            if rec['_au'] not in au_uids:
                continue
            extents[rec['_au']] = gws.gis.extent.from_box(rec['p'])

    return extents


def _enum_layers(action, au_uids):
    layers = {}
    images = _enum_images(action)

    au_index = {au.uid: au for au in action.au_list}
    type_index = {ty.uid: ty for ty in action.type_list}

    def _layer_uid(rec, geom_type):
        return rec['_type'].lower() + '_' + geom_type + '_' + rec['_au']

    def _new_layer(rec, geom_type):
        au = au_index.get(rec['_au'])
        au_name = au.name if au else ''

        ty = type_index.get(rec['_type'])
        type_name = ty.name if ty else ''
        color = ty.color if ty else ''

        return {
            'uid': _layer_uid(rec, geom_type),
            'geom': geom_type,
            'type': rec['_type'],
            'type_name': type_name,
            'au_uid': rec['_au'],
            'au_name': au_name,
            'color': color,
            'images': [],
        }

    with action.db.connect() as conn:

        tab = conn.quote_table(action.plan_table.name)
        rs = conn.select(f'SELECT _uid, _au, _type, _geom_p, _geom_l, _geom_x FROM {tab} ORDER BY _uid')

        for rec in rs:
            if rec['_au'] not in au_uids:
                continue

            for g in 'plx':
                if rec['_geom_' + g]:
                    layer_uid = _layer_uid(rec, g)
                    if layer_uid not in layers:
                        layers[layer_uid] = _new_layer(rec, g)

            imgs = [img for img in images if img['fname'].startswith(rec['_uid'])]

            if imgs:
                layer_uid = _layer_uid(rec, 'r')
                if layer_uid not in layers:
                    layers[layer_uid] = _new_layer(rec, 'r')
                layers[layer_uid]['images'].extend(imgs)

    ls = sorted(layers.values(), key=lambda la: la['type_name'])

    for la in ls:
        la['images'].sort(key=lambda img: img['fname'], reverse=True)

    return ls


def _enum_images(action):
    dd = action.data_dir
    images = []

    for path in os2.find_files(f'{dd}/png', ext='png'):
        fn = _fnbody(path)
        converted_path = f'{dd}/cnv/{fn}.png'

        if os2.file_mtime(converted_path) < os2.file_mtime(path):
            try:
                # reduce the image palette (20-30 colors work just fine for scanned plans)
                gws.log.debug(f'converting {path!r}')
                img = PIL.Image.open(path)
                img = img.convert('RGBA')
                img = img.convert('P', palette=PIL.Image.ADAPTIVE, colors=action.image_quality)
                img.save(converted_path)
                os2.chown(converted_path)

                # copy the pgw along
                pgw = gws.read_file(f'{dd}/png/{fn}.pgw')
                gws.write_file(f'{dd}/cnv/{fn}.pgw', pgw)
            except Exception as e:
                gws.log.error(f'error converting {path!r}: {e}')
                continue

        try:
            palette = _image_palette(converted_path)
        except Exception as e:
            gws.log.error(f'error getting palette from {converted_path!r}: {e}')
            continue

        images.append({
            'uid': '_r_' + fn,
            'fname': fn,
            'path': converted_path,
            'palette': palette
        })

    return images


def _image_palette(path):
    colors = []
    img = PIL.Image.open(path)
    palette = img.getpalette()

    # transparency is either a 256-bytes array for each entry or an integer index
    transparency = img.info.get('transparency', None)
    if isinstance(transparency, bytes) and len(transparency) < 256:
        transparency = None

    for n in range(255):
        r = palette[n * 3 + 0]
        g = palette[n * 3 + 1]
        b = palette[n * 3 + 2]

        if isinstance(transparency, int):
            alpha = 0 if n == transparency else 0xFF
        elif isinstance(transparency, bytes):
            alpha = transparency[n]
        else:
            alpha = 0xFF

        colors.append(['#%02x%02x%02x' % (r, g, b), alpha])

    return colors


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
    for a in au_uids:
        if fn.startswith(a) or (a + '_' in fn) or ('_' + a in fn):
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


def _delete_feature_assets(dir, uid):
    for p in os2.find_files(dir, ext=['pdf', 'png', 'pgw']):
        fn = _filename(p)
        if fn.startswith(uid):
            gws.log.debug(f'DELETE {p}')
            os2.unlink(p)
