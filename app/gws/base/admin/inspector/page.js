const C = JSON.parse(document.getElementById('config').textContent);
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

function renderResults() {
    const box = document.getElementById('nodes');
    const n = C.results.length;
    box.append(el('div', { class: 'count', text: n >= C.maxResults ? `${n}+ found` : `${n} found` }));
    for (const r of C.results) {
        const a = el('a', { class: 'node', href: href(r.path) }, el('div', { class: 'uid', text: r.label }));
        for (const m of r.matches) {
            a.append(el('div', { class: 'info', text: `${m.key}=${m.value}` }));
        }
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

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.getElementById('search').classList.contains('searching')) {
            window.stop();
            setSearching(false);
        }
    });
    window.addEventListener('pageshow', () => setSearching(false));
}

function renderCrumbs() {
    const box = document.getElementById('crumbs');
    C.crumbs.forEach((c, n) => {
        if (n > 0) box.append(el('span', { class: 'sep', text: '/' }));
        box.append(el('a', { href: href(c.path), text: c.label }));
    });
}

function renderValue(v) {
    switch (v.kind) {
        case 'primitive':
            return el('span', { class: `primitive t-${v.baseType}`, text: v.value });
        case 'object':
            return el('a', { href: href(v.path), text: v.label });
        case 'collection': {
            const d = el('details', {}, el('summary', { text: v.label }));
            if (v.items.length) d.append(renderTable(v.items));
            return d;
        }
        default:
            return el('span', { class: 'other', text: v.value });
    }
}

function renderTable(props) {
    const table = el('table', { class: 'props' });
    for (const p of props) {
        table.append(el('tr', {}, el('td', { class: 'key', text: p.key }), el('td', { class: 'value' }, renderValue(p))));
    }
    return table;
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
}

main();
