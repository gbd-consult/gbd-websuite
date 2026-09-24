const C = JSON.parse(document.getElementById('config').textContent);
const MAX_GRID_CELLS = 2000;

const EYE_ICON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF_ICON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>' +
    '<path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>' +
    '<path d="M14.12 14.12a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';

let map = null;
let S = null;
const layers = {};
const visible = { cache: true, grid: true, map: true };

function tilePath(z, x, y) {
    const s = 10000;
    const pad = (n, w) => String(n).padStart(w, '0');
    return `${pad(z, 2)}/${pad(Math.floor(x / s), 4)}/${pad(x % s, 4)}/${pad(Math.floor(y / s), 4)}/${pad(y % s, 4)}.${S.ext}`;
}

function tileUrl(tileCoord) {
    const x = tileCoord[1];
    const y = -tileCoord[2] - 1;
    const r = S.range;
    if (!r || x < r[0] || x > r[2] || y < r[1] || y > r[3]) return undefined;
    return `${C.url}${S.name}/${S.z}/${x}/${y}.${S.ext}`;
}

function select(name, z) {
    const cache = C.caches[name];
    if (!cache) return;
    S = {
        name,
        z,
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

    const [total, cached] = cache.counts[z];
    const pct = total ? Math.floor((100 * cached) / total) : 0;
    document.getElementById('status').textContent = `cache ${name} | level=${z} | total=${total} | cached=${cached} (${pct}%)`;

    for (const el of document.querySelectorAll('.cache, .level')) {
        const on = el.dataset.name === name && (el.classList.contains('cache') || el.dataset.z === String(z));
        el.classList.toggle('selected', on);
    }

    showMap();
}

function rangeExtent() {
    const r = S.range;
    if (!r) return S.extent;
    const span = S.resolution * S.tileSize;
    const [ox, , , oy] = S.gridExtent;
    return [ox + r[0] * span, oy - (r[3] + 1) * span, ox + (r[2] + 1) * span, oy - r[1] * span];
}

function showMap() {
    if (map) {
        map.setTarget(null);
        map = null;
    }

    layers.map = new ol.layer.Tile({ source: new ol.source.OSM({ wrapX: false }), opacity: 0.3 });
    layers.grid = newGridLayer();
    layers.cache = null;

    if (!S) {
        map = new ol.Map({
            target: 'map',
            layers: [layers.map, layers.grid],
            view: new ol.View({ center: [0, 0], zoom: 2 }),
        });
        map.getView().on('change:resolution', showZoom);
        map.on('moveend', drawGrid);
        applyVisibility();
        showZoom();
        return;
    }

    const code = S.crs.epsg;
    if (ol.proj.setProj4) ol.proj.setProj4(proj4);
    if (!ol.proj.get(code) && S.crs.proj4text) proj4.defs(code, S.crs.proj4text);
    const proj = ol.proj.get(code);
    if (!proj) {
        document.getElementById('status').textContent = `unknown projection ${code}`;
        return;
    }
    proj.setExtent(S.gridExtent);

    const grid = new ol.tilegrid.TileGrid({
        extent: S.gridExtent,
        origin: [S.gridExtent[0], S.gridExtent[3]],
        resolutions: [S.resolution],
        tileSize: S.tileSize,
    });

    const source = new ol.source.XYZ({
        projection: proj,
        tileGrid: grid,
        tileUrlFunction: tileUrl,
        wrapX: false,
    });

    layers.cache = new ol.layer.Tile({ source });

    map = new ol.Map({
        target: 'map',
        layers: [layers.map, layers.cache, layers.grid],
        view: new ol.View({ projection: proj, extent: S.gridExtent }),
    });
    applyVisibility();

    map.getView().on('change:resolution', showZoom);
    map.on('moveend', drawGrid);
    map.getView().setCenter(ol.extent.getCenter(rangeExtent()));
    map.getView().setResolution(S.resolution);
    showZoom();
}

function newGridLayer() {
    const stroke = new ol.style.Stroke({ color: 'rgba(0, 0, 200, 0.6)', width: 1 });
    return new ol.layer.Vector({
        source: new ol.source.Vector(),
        style: (feature) => {
            if (feature.get('highlight')) {
                return new ol.style.Style({ fill: new ol.style.Fill({ color: 'rgba(255, 165, 0, 0.15)' }) });
            }
            return new ol.style.Style({
                stroke,
                text: new ol.style.Text({
                    text: feature.get('label'),
                    font: '12px sans-serif',
                    fill: new ol.style.Fill({ color: 'rgb(0, 0, 200)' }),
                    stroke: new ol.style.Stroke({ color: 'white', width: 3 }),
                }),
            });
        },
    });
}

function levelForResolution(g, res) {
    for (let z = 0; z < 100; z++) {
        if (g.baseResolution / 2 ** z <= res * 1.01) return z;
    }
    return 99;
}

function drawGrid() {
    const src = layers.grid.getSource();
    src.clear();

    const g = S ? S.grid : C.baseGrid;
    const z = S ? S.z : levelForResolution(g, map.getView().getResolution());
    const span = (g.baseResolution / 2 ** z) * g.tileSize;
    const [ox, , , oy] = g.extent;
    const cellExtent = (x0, y0, x1, y1) => [ox + x0 * span, oy - (y1 + 1) * span, ox + (x1 + 1) * span, oy - y0 * span];

    if (S) {
        const [gx0, gy0, gx1, gy1] = S.gridRange;
        const h = new ol.Feature(ol.geom.Polygon.fromExtent(cellExtent(gx0, gy0, gx1, gy1)));
        h.set('highlight', true);
        src.addFeature(h);
    }

    const e = ol.extent.getIntersection(map.getView().calculateExtent(map.getSize()), g.extent);
    if (ol.extent.isEmpty(e)) return;

    const x0 = Math.floor((e[0] - ox) / span);
    const x1 = Math.ceil((e[2] - ox) / span) - 1;
    const y0 = Math.floor((oy - e[3]) / span);
    const y1 = Math.ceil((oy - e[1]) / span) - 1;
    if ((x1 - x0 + 1) * (y1 - y0 + 1) > MAX_GRID_CELLS) return;

    const features = [];
    for (let x = x0; x <= x1; x++) {
        for (let y = y0; y <= y1; y++) {
            const f = new ol.Feature(ol.geom.Polygon.fromExtent(cellExtent(x, y, x, y)));
            f.set('label', `${x},${y} - ${z}`);
            features.push(f);
        }
    }
    src.addFeatures(features);
}

function applyVisibility() {
    for (const [name, layer] of Object.entries(layers)) {
        if (layer) layer.setVisible(visible[name]);
    }
    for (const btn of document.querySelectorAll('#toolbar .toggle')) {
        const on = visible[btn.dataset.layer];
        btn.classList.toggle('off', !on);
        btn.innerHTML = (on ? EYE_ICON : EYE_OFF_ICON) + btn.dataset.layer;
    }
}

function showZoom() {
    const view = map.getView();
    document.getElementById('zoom').textContent = `${Math.floor(view.getZoom())} (${view.getResolution().toFixed(4)})`;
}

function filter() {
    const q = document.getElementById('search-input').value.toLowerCase();
    for (const el of document.querySelectorAll('.cache')) {
        const on = !q || el.textContent.toLowerCase().includes(q);
        el.classList.toggle('hidden', !on);
    }
}

function main() {
    document.getElementById('search-input').addEventListener('input', filter);
    document.getElementById('search-clear').addEventListener('click', () => {
        document.getElementById('search-input').value = '';
        filter();
    });

    for (const btn of document.querySelectorAll('#toolbar .toggle')) {
        btn.addEventListener('click', () => {
            visible[btn.dataset.layer] = !visible[btn.dataset.layer];
            applyVisibility();
        });
    }

    document.getElementById('sidebar').addEventListener('click', (evt) => {
        const a = evt.target.closest('a.level');
        if (!a) return;
        evt.preventDefault();
        select(a.dataset.name, parseInt(a.dataset.z, 10));
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
