const C = JSON.parse(document.getElementById('config').textContent);
const Q = parseSearch(C.search);

function parseSearch(s) {
    const m = s.match(/^([^=\s]+)\s*=(.*)$/);
    const [prop, text] = m ? [m[1], m[2].trim()] : [null, s];
    return { prop, text: text.toLowerCase() };
}

function href(path, search = C.search) {
    let s = C.url + encodeURIComponent(path).replace(/%2F/g, '/');
    if (search) s += '&search=' + encodeURIComponent(search);
    return s;
}

function el(tag, attrs, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
        if (k === 'text') e.textContent = v;
        else e.setAttribute(k, v);
    }
    for (const c of children) {
        if (c) e.append(c);
    }
    return e;
}

function renderNodes() {
    const box = document.getElementById('nodes');
    for (const n of C.nodes) {
        const a = el('a', { class: 'node', href: href(n.path) }, el('div', { class: 'uid', text: n.label }));
        if (n.uid === C.selectedUid) a.classList.add('selected');
        box.append(a);
    }
}

function renderPath(box, crumbs) {
    crumbs.forEach((c, n) => {
        if (n > 0) box.append(el('span', { class: 'sep', text: '.' }));
        box.append(el('a', { href: href(c.path), text: c.label }));
    });
}

function renderResults() {
    const box = document.getElementById('nodes');
    const n = C.results.length;
    box.append(el('div', { class: 'count', text: n >= C.maxResults ? `${n}+ found` : `${n} found` }));
    for (const r of C.results) {
        const a = el('a', { class: 'node result', href: href(r.path), text: r.label });
        if (r.path === C.path) a.classList.add('selected');
        box.append(a);
    }
}

function setSearching(on) {
    document.getElementById('search').classList.toggle('searching', on);
}

function initSearch() {
    const input = document.getElementById('search-input');
    input.value = C.search;

    const start = () => {
        setSearching(true);
        location.href = href(C.path, input.value.trim());
    };

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') start();
    });
    document.getElementById('search-button').addEventListener('click', start);
    document.getElementById('search-clear').addEventListener('click', () => {
        input.value = '';
        if (C.search) start();
        else input.focus();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.getElementById('search').classList.contains('searching')) {
            window.stop();
            setSearching(false);
        }
    });
    window.addEventListener('pageshow', () => setSearching(false));
}

function renderCrumbs() {
    renderPath(document.getElementById('crumbs'), C.crumbs);
}

function renderPrimitive(v, key, parentKey) {
    const span = el('span', { class: `primitive t-${v.baseType}` });
    const s = v.value;
    const t = Q.text;
    if (!t || (Q.prop !== null && key !== Q.prop && parentKey !== Q.prop)) {
        span.textContent = s;
        return span;
    }
    const low = s.toLowerCase();
    let pos = 0;
    while (true) {
        const i = low.indexOf(t, pos);
        if (i < 0) break;
        span.append(s.slice(pos, i), el('mark', { text: s.slice(i, i + t.length) }));
        pos = i + t.length;
    }
    span.append(s.slice(pos));
    return span;
}

function renderValue(v, key, parentKey) {
    switch (v.kind) {
        case 'primitive':
            return renderPrimitive(v, key, parentKey);
        case 'object':
            return el('a', { href: href(v.path), text: v.label });
        case 'collection': {
            if (!v.items.length) {
                const s = el('summary', { text: v.label });
                s.addEventListener('click', (e) => e.preventDefault());
                return el('details', { class: 'empty' }, s);
            }
            return el('details', {}, el('summary', { text: v.label }), renderTable(v.items, key));
        }
        default:
            return el('span', { class: 'other', text: v.value });
    }
}

function renderTable(props, parentKey) {
    const table = el('table', { class: 'props' });
    for (const p of props) {
        table.append(el('tr', {}, el('td', { class: 'key', text: p.key }), el('td', { class: 'value' }, renderValue(p, p.key, parentKey))));
    }
    return table;
}

function revealMarks() {
    const marks = document.querySelectorAll('#props mark');
    for (const m of marks) {
        for (let d = m.closest('details'); d; d = d.parentElement.closest('details')) {
            d.open = true;
        }
    }
    if (marks.length) marks[0].scrollIntoView({ block: 'center' });
}

function main() {
    if (C.results) renderResults();
    else renderNodes();
    initSearch();
    renderCrumbs();
    document.getElementById('label').textContent = C.label;
    document.getElementById('props').append(renderTable(C.props));
    const sel = document.querySelector('.node.selected');
    if (sel) sel.scrollIntoView({ block: 'center' });
    revealMarks();
}

main();
