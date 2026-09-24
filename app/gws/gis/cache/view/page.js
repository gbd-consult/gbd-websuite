var C = JSON.parse(document.getElementById('config').textContent);

function pad(n, w) {
    var s = String(n);
    while (s.length < w) s = '0' + s;
    return s;
}

function tilePath(z, x, y) {
    var s = 10000;
    return (
        pad(z, 2) +
        '/' +
        pad(Math.floor(x / s), 4) +
        '/' +
        pad(x % s, 4) +
        '/' +
        pad(Math.floor(y / s), 4) +
        '/' +
        pad(y % s, 4) +
        '.' +
        S.ext
    );
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
var MAX_GRID_CELLS = 2000;

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
        gridRange: cache.gridRanges[z],
        ext: cache.ext,
        grid: cache.grid,
    };

    var total = cache.counts[z][0],
        cached = cache.counts[z][1];
    var pct = total ? Math.floor((100 * cached) / total) : 0;
    document.getElementById('status').textContent = [
        'cache ',
        name,
        '/',
        cache.srid,
        ' | level=',
        z,
        ' | total=',
        total,
        ' | cached=',
        cached,
        ' (',
        pct,
        '%)',
    ].join('');

    var els = document.querySelectorAll('.cache, .level');
    for (var i = 0; i < els.length; i++) {
        var el = els[i];
        var on =
            el.getAttribute('data-name') === name &&
            (el.className.indexOf('cache') >= 0 || el.getAttribute('data-z') === String(z));
        el.className = el.className.replace(/ ?\bselected\b/, '') + (on ? ' selected' : '');
    }

    showMap();
}

function rangeExtent() {
    var r = S.range;
    if (!r) return S.extent;
    var span = S.resolution * S.tileSize;
    var ox = S.gridExtent[0],
        oy = S.gridExtent[3];
    return [ox + r[0] * span, oy - (r[3] + 1) * span, ox + (r[2] + 1) * span, oy - r[1] * span];
}

function showMap() {
    if (map) {
        map.setTarget(null);
        map = null;
    }

    var base = new ol.layer.Tile({ source: new ol.source.OSM({ wrapX: false }), opacity: 0.3 });

    if (!S) {
        map = new ol.Map({
            target: 'map',
            layers: [base, newGridLayer()],
            view: new ol.View({ center: [0, 0], zoom: 2 }),
        });
        map.getView().on('change:resolution', showZoom);
        map.on('moveend', drawGrid);
        showZoom();
        return;
    }

    var code = S.crs.epsg;
    if (ol.proj.setProj4) ol.proj.setProj4(proj4);
    if (!ol.proj.get(code) && S.crs.proj4text) proj4.defs(code, S.crs.proj4text);
    var proj = ol.proj.get(code);
    if (!proj) {
        document.getElementById('status').textContent = 'unknown projection ' + code;
        return;
    }
    proj.setExtent(S.gridExtent);

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
        layers: [base, new ol.layer.Tile({ source: source }), newGridLayer()],
        view: new ol.View({ projection: proj, extent: S.gridExtent }),
    });

    map.getView().on('change:resolution', showZoom);
    map.on('moveend', drawGrid);
    map.getView().setCenter(ol.extent.getCenter(rangeExtent()));
    map.getView().setResolution(S.resolution);
    showZoom();
}

var gridLayer = null;

function newGridLayer() {
    var stroke = new ol.style.Stroke({ color: 'rgba(0, 0, 200, 0.6)', width: 1 });
    gridLayer = new ol.layer.Vector({
        source: new ol.source.Vector(),
        style: function (feature) {
            if (feature.get('highlight')) {
                return new ol.style.Style({ fill: new ol.style.Fill({ color: 'rgba(255, 165, 0, 0.15)' }) });
            }
            return new ol.style.Style({
                stroke: stroke,
                text: new ol.style.Text({
                    text: feature.get('label'),
                    font: '12px sans-serif',
                    fill: new ol.style.Fill({ color: 'rgb(0, 0, 200)' }),
                    stroke: new ol.style.Stroke({ color: 'white', width: 3 }),
                }),
            });
        },
    });
    return gridLayer;
}

function levelForResolution(g, res) {
    for (var z = 0; z < 100; z++) {
        var r = g.baseResolution / Math.pow(2, z);
        if (r <= res * 1.01) return z;
    }
    return 99;
}

function drawGrid() {
    var src = gridLayer.getSource();
    src.clear();

    var g = S ? S.grid : C.baseGrid;
    var z = S ? S.z : levelForResolution(g, map.getView().getResolution());
    var span = (g.baseResolution / Math.pow(2, z)) * g.tileSize;
    var ox = g.extent[0],
        oy = g.extent[3];

    if (S) {
        var gr = S.gridRange;
        var h = new ol.Feature(
            ol.geom.Polygon.fromExtent([ox + gr[0] * span, oy - (gr[3] + 1) * span, ox + (gr[2] + 1) * span, oy - gr[1] * span])
        );
        h.set('highlight', true);
        src.addFeature(h);
    }

    var e = ol.extent.getIntersection(map.getView().calculateExtent(map.getSize()), g.extent);
    if (ol.extent.isEmpty(e)) return;

    var x0 = Math.floor((e[0] - ox) / span),
        x1 = Math.ceil((e[2] - ox) / span) - 1,
        y0 = Math.floor((oy - e[3]) / span),
        y1 = Math.ceil((oy - e[1]) / span) - 1;
    if ((x1 - x0 + 1) * (y1 - y0 + 1) > MAX_GRID_CELLS) return;

    var features = [];
    for (var x = x0; x <= x1; x++) {
        for (var y = y0; y <= y1; y++) {
            var f = new ol.Feature(
                ol.geom.Polygon.fromExtent([ox + x * span, oy - (y + 1) * span, ox + (x + 1) * span, oy - y * span])
            );
            f.set('label', x + ',' + y + ' - ' + z);
            features.push(f);
        }
    }
    src.addFeatures(features);
}

function showZoom() {
    var view = map.getView();
    document.getElementById('zoom').textContent = Math.floor(view.getZoom()) + ' (' + view.getResolution().toFixed(4) + ')';
}

function filter() {
    var q = document.getElementById('search-input').value.toLowerCase();
    var els = document.querySelectorAll('.cache');
    for (var i = 0; i < els.length; i++) {
        var el = els[i];
        var on = !q || el.textContent.toLowerCase().indexOf(q) >= 0;
        el.className = el.className.replace(/ ?\bhidden\b/, '') + (on ? '' : ' hidden');
    }
}

function main() {
    document.getElementById('search-input').addEventListener('input', filter);
    document.getElementById('search-clear').addEventListener('click', function () {
        document.getElementById('search-input').value = '';
        filter();
    });

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
        showMap();
    }
}

main();
