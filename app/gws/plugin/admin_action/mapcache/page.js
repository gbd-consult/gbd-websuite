const C = JSON.parse(document.getElementById('config').textContent);
const MAX_GRID_CELLS = 2000;
const MAX_CACHE_TILES = 512;
const PAGE_SIZE = 100;

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
let activeFilter = 'all';
let visibleLimit = PAGE_SIZE;

const numberFormatter = new Intl.NumberFormat(document.documentElement.lang || 'en');

function formatNumber(value) {
    return numberFormatter.format(Number(value));
}

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
    const cacheEl = [...document.querySelectorAll('.cache')].find((el) => el.dataset.name === name);
    const levelEl = cacheEl && cacheEl.querySelector(`.level[data-z="${z}"]`);
    const pct = levelEl ? Number(levelEl.dataset.percent) : total ? Math.floor((100 * cached) / total) : 0;
    const title = cacheEl ? cacheEl.dataset.title : name;
    const resolution = levelEl ? Number(levelEl.dataset.resolution) : S.resolution;

    document.querySelector('#status .status-empty').hidden = true;
    document.querySelector('#status .status-content').hidden = false;
    document.getElementById('status-cache-name').textContent = title || name;
    document.getElementById('status-cache-name').title = name;
    document.getElementById('status-level').textContent = `Zoom level ${z}`;
    document.getElementById('status-total').textContent = formatNumber(total);
    document.getElementById('status-cached').textContent = formatNumber(cached);
    document.getElementById('status-percent').textContent = `${pct}%`;
    document.getElementById('status-resolution').textContent = `${resolution.toFixed(4)} m/px`;

    for (const el of document.querySelectorAll('.cache, .level')) {
        const on = el.dataset.name === name && (el.classList.contains('cache') || el.dataset.z === String(z));
        el.classList.toggle('selected', on);
    }

    if (cacheEl) {
        cacheEl.classList.add('open');
        cacheEl.querySelector('.cache-toggle').setAttribute('aria-expanded', 'true');
        const details = cacheEl.querySelector('.cache-levels');
        details.inert = false;
        details.setAttribute('aria-hidden', 'false');
        cacheEl.scrollIntoView({ block: 'nearest' });
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

function cacheTilesInView() {
    if (!S || !map) return 0;

    const size = map.getSize();
    const resolution = map.getView().getResolution();
    if (!size || !resolution) return 0;

    const e = ol.extent.getIntersection(map.getView().calculateExtent(size), S.gridExtent);
    if (ol.extent.isEmpty(e)) return 0;

    const span = S.resolution * S.tileSize;
    const columns = Math.ceil((e[2] - e[0]) / span) + 1;
    const rows = Math.ceil((e[3] - e[1]) / span) + 1;
    return columns * rows;
}

function cacheIsLimited() {
    return cacheTilesInView() > MAX_CACHE_TILES;
}

function closestCachedLevel() {
    if (!S || !map) return null;
    const cache = C.caches[S.name];
    const viewZ = Math.round(map.getView().getZoom());
    const levels = Object.entries(cache.counts)
        .filter(([z, counts]) => counts[1] > 0 && cache.resolutions[z] !== undefined)
        .map(([z]) => Number(z));
    if (!levels.length) return null;
    return levels.reduce((best, z) => {
        const distance = Math.abs(z - viewZ);
        const bestDistance = Math.abs(best - viewZ);
        return distance < bestDistance ? z : best;
    });
}

function showMap() {
    if (map) {
        map.setTarget(null);
        map = null;
    }

    layers.map = new ol.layer.Tile({ source: new ol.source.OSM({ wrapX: false }), opacity: 0.9 });
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
        const empty = document.querySelector('#status .status-empty');
        empty.hidden = false;
        empty.textContent = `Unknown projection ${code}`;
        document.querySelector('#status .status-content').hidden = true;
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

    layers.cache = new ol.layer.Tile({ source, opacity: 0.22 });

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
    const cacheExtentStroke = new ol.style.Stroke({ color: 'rgba(36, 87, 214, 0.8)', width: 2, lineDash: [7, 5] });
    return new ol.layer.Vector({
        source: new ol.source.Vector(),
        style: (feature) => {
            if (feature.get('highlight')) {
                return new ol.style.Style({
                    fill: new ol.style.Fill({ color: 'rgba(36, 87, 214, 0.025)' }),
                    stroke: cacheExtentStroke,
                });
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
        const limited = name === 'cache' && cacheIsLimited();
        if (layer) layer.setVisible(visible[name] && !limited);
    }
    for (const btn of document.querySelectorAll('#toolbar .toggle')) {
        const unavailable = btn.dataset.layer === 'cache' && !layers.cache;
        const limited = btn.dataset.layer === 'cache' && cacheIsLimited();
        const on = visible[btn.dataset.layer] && !limited && !unavailable;
        if (!btn.dataset.defaultTitle) btn.dataset.defaultTitle = btn.title;
        const state = `${on}:${limited}:${unavailable}`;
        if (btn.dataset.state === state) continue;
        btn.dataset.state = state;
        btn.classList.toggle('off', !on);
        btn.classList.toggle('limited', limited);
        btn.classList.toggle('unavailable', unavailable);
        btn.disabled = limited || unavailable;
        btn.setAttribute('aria-pressed', String(on));
        btn.title = unavailable
            ? 'Select a cache level to enable this layer'
            : limited
                ? 'Cache hidden at this zoom to keep the map responsive'
                : btn.dataset.defaultTitle;
        const label = btn.querySelector('.toggle-copy');
        btn.innerHTML = on ? EYE_ICON : EYE_OFF_ICON;
        if (label) btn.appendChild(label);
    }
}

function showZoom() {
    const view = map.getView();
    const tileCount = cacheTilesInView();
    const limited = tileCount > MAX_CACHE_TILES;
    const warning = document.getElementById('performance-warning');
    document.getElementById('zoom-value').textContent = `z ${Math.floor(view.getZoom())}`;
    document.getElementById('zoom-resolution').textContent = `${view.getResolution().toFixed(4)} m/px`;
    warning.hidden = !limited;
    if (limited) {
        const suggestedZ = closestCachedLevel();
        const action = document.getElementById('zoom-to-level');
        if (suggestedZ === null) {
            delete warning.dataset.z;
            document.getElementById('performance-warning-hint').textContent = 'No cached level is available';
            action.hidden = true;
        } else {
            warning.dataset.z = suggestedZ;
            document.getElementById('performance-warning-hint').textContent = `Closest cached level is z${suggestedZ}`;
            action.textContent = `Use z${suggestedZ}`;
            action.hidden = false;
        }
    }
    applyVisibility();
}

function initializeCacheList() {
    const cacheList = document.getElementById('caches');
    const selected = cacheList.querySelector('.cache.selected');
    if (selected) cacheList.prepend(selected);

    for (const cacheEl of cacheList.querySelectorAll('.cache')) {
        const details = cacheEl.querySelector('.cache-levels');
        const isOpen = cacheEl.classList.contains('open');
        details.inert = !isOpen;
        details.setAttribute('aria-hidden', String(!isOpen));
        const levels = [...cacheEl.querySelectorAll('.level')];
        const cachedLevels = levels.filter((level) => Number(level.dataset.cached) > 0).length;
        const completeLevels = levels.filter((level) => Number(level.dataset.percent) === 100).length;
        const state = completeLevels > 0 ? 'complete' : cachedLevels > 0 ? 'partial' : 'empty';
        const stateLabel = state === 'complete' ? `${completeLevels} complete` : state === 'partial' ? 'Has tiles' : 'Empty';
        cacheEl.dataset.state = state;
        cacheEl.querySelector('.cache-state').textContent = stateLabel;
        const meta = cacheEl.querySelector('.cache-meta');
        meta.append(` · ${cachedLevels}/${levels.length} cached levels`);
        meta.title = meta.textContent;
    }
}

function filterCaches(resetLimit = true) {
    if (resetLimit) visibleLimit = PAGE_SIZE;
    const input = document.getElementById('search-input');
    const q = input.value.trim().toLowerCase();
    const caches = [...document.querySelectorAll('.cache')];
    const matches = caches.filter((cacheEl) => {
        const textMatches = !q || `${cacheEl.dataset.name} ${cacheEl.dataset.title}`.toLowerCase().includes(q);
        const state = cacheEl.dataset.state;
        const stateMatches = activeFilter === 'all'
            || (activeFilter === 'cached' && state !== 'empty')
            || state === activeFilter;
        return textMatches && stateMatches;
    });

    const matchSet = new Set(matches);
    const shown = new Set(matches.slice(0, visibleLimit));
    for (const cacheEl of caches) {
        const match = matchSet.has(cacheEl);
        cacheEl.classList.toggle('filtered-out', !match);
        cacheEl.classList.toggle('page-hidden', match && !shown.has(cacheEl));
    }

    document.getElementById('search').classList.toggle('has-value', Boolean(input.value));
    document.getElementById('result-count').textContent = `${formatNumber(matches.length)} result${matches.length === 1 ? '' : 's'}`;
    document.getElementById('empty-state').hidden = matches.length !== 0;

    const loadMore = document.getElementById('load-more');
    const remaining = Math.max(0, matches.length - shown.size);
    loadMore.hidden = remaining === 0;
    loadMore.textContent = `Show ${formatNumber(Math.min(PAGE_SIZE, remaining))} more`;
}

function toggleCache(cacheEl) {
    const open = !cacheEl.classList.contains('open');
    cacheEl.classList.toggle('open', open);
    cacheEl.querySelector('.cache-toggle').setAttribute('aria-expanded', String(open));
    const details = cacheEl.querySelector('.cache-levels');
    details.inert = !open;
    details.setAttribute('aria-hidden', String(!open));
}

function main() {
    initializeCacheList();

    document.getElementById('search-input').addEventListener('input', () => filterCaches());
    document.getElementById('search-clear').addEventListener('click', () => {
        document.getElementById('search-input').value = '';
        document.getElementById('search-input').focus();
        filterCaches();
    });

    for (const btn of document.querySelectorAll('#cache-filters button')) {
        btn.addEventListener('click', () => {
            activeFilter = btn.dataset.filter;
            for (const filterBtn of document.querySelectorAll('#cache-filters button')) {
                filterBtn.setAttribute('aria-pressed', String(filterBtn === btn));
            }
            filterCaches();
        });
    }

    document.getElementById('load-more').addEventListener('click', () => {
        visibleLimit += PAGE_SIZE;
        filterCaches(false);
    });

    document.getElementById('sidebar-toggle').addEventListener('click', () => {
        const collapsed = document.body.classList.toggle('sidebar-collapsed');
        const toggle = document.getElementById('sidebar-toggle');
        const sidebar = document.getElementById('sidebar');
        const label = collapsed ? 'Expand cache list' : 'Collapse cache list';
        toggle.setAttribute('aria-expanded', String(!collapsed));
        toggle.title = label;
        toggle.querySelector('.sr-only').textContent = label;
        sidebar.inert = collapsed;
        sidebar.setAttribute('aria-hidden', String(collapsed));
        if (map) setTimeout(() => map.updateSize(), 190);
    });

    for (const btn of document.querySelectorAll('#toolbar .toggle')) {
        btn.addEventListener('click', () => {
            visible[btn.dataset.layer] = !visible[btn.dataset.layer];
            applyVisibility();
        });
    }

    document.getElementById('sidebar').addEventListener('click', (evt) => {
        const a = evt.target.closest('a.level');
        if (a) {
            evt.preventDefault();
            select(a.dataset.name, parseInt(a.dataset.z, 10));
            history.replaceState(null, '', a.getAttribute('href'));
            return;
        }

        const toggle = evt.target.closest('.cache-toggle');
        if (toggle) toggleCache(toggle.closest('.cache'));
    });

    document.getElementById('zoom-to-level').addEventListener('click', () => {
        if (!map || !S) return;
        const name = S.name;
        const z = Number(document.getElementById('performance-warning').dataset.z);
        if (!Number.isInteger(z) || !C.caches[name].counts[z]) return;
        select(name, z);
        history.replaceState(null, '', `${C.url}${name}/${z}`);
    });

    filterCaches();

    if (C.selected) {
        select(C.selected.name, C.selected.z);
    } else {
        showMap();
    }
}

main();
