var C = JSON.parse(document.getElementById('config').textContent);

function pad(n, w) {
    var s = String(n);
    while (s.length < w) s = '0' + s;
    return s;
}

function tilePath(z, x, y) {
    var s = 10000;
    return pad(z, 2) + '/' + pad(Math.floor(x / s), 4) + '/' + pad(x % s, 4) + '/' + pad(Math.floor(y / s), 4) + '/' + pad(y % s, 4) + '.' + S.ext;
}

function tileUrl(tileCoord) {
    var x = tileCoord[1];
    var y = -tileCoord[2] - 1;
    var r = S.range;
    if (!r || x < r[0] || x > r[2] || y < r[1] || y > r[3]) return undefined;
    return C.url + S.name + '/' + S.z + '/' + x + '/' + y + '.' + S.ext;
}

var map = null;
var S = null;

function select(name, z) {
    var cache = C.caches[name];
    if (!cache) return;
    S = {
        name: name,
        z: z,
        crs: cache.crs,
        gridExtent: cache.gridExtent,
        tileSize: cache.tileSize,
        resolution: cache.resolutions[z],
        extent: cache.extent,
        range: cache.ranges[z],
        ext: cache.ext,
    };

    var total = cache.counts[z][0], cached = cache.counts[z][1];
    var pct = total ? Math.floor(100 * cached / total) : 0;
    document.getElementById('status').textContent = name + ' / ' + cache.srid + ' :: level ' + z + ' :: total ' + total + ' :: cached ' + cached + ' (' + pct + '%)';

    var els = document.querySelectorAll('.cache, .level');
    for (var i = 0; i < els.length; i++) {
        var el = els[i];
        var on = el.getAttribute('data-name') === name && (el.className.indexOf('cache') >= 0 || el.getAttribute('data-z') === String(z));
        el.className = el.className.replace(/ ?\bselected\b/, '') + (on ? ' selected' : '');
    }

    showMap();
}

function rangeExtent() {
    var r = S.range;
    if (!r) return S.extent;
    var span = S.resolution * S.tileSize;
    var ox = S.gridExtent[0], oy = S.gridExtent[3];
    return [ox + r[0] * span, oy - (r[3] + 1) * span, ox + (r[2] + 1) * span, oy - r[1] * span];
}

function showMap() {
    if (map) {
        map.setTarget(null);
        map = null;
    }

    var code = S.crs.epsg;
    if (ol.proj.setProj4) ol.proj.setProj4(proj4);
    if (!ol.proj.get(code) && S.crs.proj4text) proj4.defs(code, S.crs.proj4text);
    var proj = ol.proj.get(code);
    if (!proj) {
        document.getElementById('status').textContent = 'unknown projection ' + code;
        return;
    }
    proj.setExtent(S.crs.extent);

    var grid = new ol.tilegrid.TileGrid({
        extent: S.gridExtent,
        origin: [S.gridExtent[0], S.gridExtent[3]],
        resolutions: [S.resolution],
        tileSize: S.tileSize,
    });

    var source = new ol.source.XYZ({
        projection: proj,
        tileGrid: grid,
        tileUrlFunction: tileUrl,
        wrapX: false,
    });

    map = new ol.Map({
        target: 'map',
        layers: [
            new ol.layer.Tile({source: new ol.source.OSM({wrapX: false}), opacity: 0.6}),
            new ol.layer.Tile({source: source}),
        ],
        view: new ol.View({projection: proj, extent: S.crs.extent}),
    });

    map.getView().fit(rangeExtent(), {constrainResolution: false});
}

function main() {
    document.getElementById('sidebar').addEventListener('click', function (evt) {
        var a = evt.target.closest('a.level');
        if (!a) return;
        evt.preventDefault();
        select(a.getAttribute('data-name'), parseInt(a.getAttribute('data-z'), 10));
        history.replaceState(null, '', a.getAttribute('href'));
    });

    if (C.selected) {
        select(C.selected.name, C.selected.z);
    } else {
        document.getElementById('status').textContent = 'select a cache level';
    }
}

main();
